import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl

class MILModel(pl.LightningModule):
    def __init__(self, hparams):
        """
        Args:
            lr: (float) initial learning rate.
            patience: (int) number of epochs without validation loss improvement
                before learning rate is decreased.
            factor: (float) factor by which learning rate is decreased.
            pos_weight: (float) weight applied to loss corresponding to positive
                samples.
        """
        super().__init__()
        self.hparams = hparams
        self.lr = self.hparams.get('lr', 1e-3)
        self.patience = self.hparams.get('patience', 10)
        self.factor = self.hparams.get('factor', 0.5)
        self.pos_weight = torch.tensor(self.hparams.get('pos_weight', 1)).float()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer,
                                      mode='min',
                                      factor=self.factor,
                                      patience=self.patience,
                                      verbose=True)
        return {'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'val_loss'}
                
    def configure_callbacks(self):
        high_BA_checkpoint = pl.callbacks.ModelCheckpoint(
            save_top_k=5,
            save_last=True,
            verbose=True,
            monitor='val_BA',
            mode='max',
        )
        return [high_BA_checkpoint]

    def training_step(self, train_batch, batch_idx):
        targets = train_batch['label']
        scores = self.forward(train_batch)
        loss = F.binary_cross_entropy_with_logits(scores,
                                                  targets,
                                                  pos_weight=self.pos_weight)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        targets = val_batch['label']
        scores = self.forward(val_batch)
        loss = F.binary_cross_entropy_with_logits(scores,
                                                  targets,
                                                  pos_weight=self.pos_weight)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        probabilities = torch.sigmoid(scores)
        return probabilities, targets

    def validation_epoch_end(self, outputs):
        probabilities = torch.cat([p for p, _ in outputs])
        targets = torch.cat([t for _, t in outputs]).int()
        metrics = self.compute_metrics(probabilities, targets)
        for metric_name, metric_value in metrics.items():
            self.log('val_'+metric_name, metric_value)
            
    @staticmethod
    def compute_metrics(probabilities, targets):
        """
        Compute a range of metrics for unbalanced classification.
        
        Args:
            probabilities: (Tensor) probabilities estimated by the model.
            targets: (Tensor) target classes.
        """
        metrics = dict()
        predictions = (probabilities >= 0.5).int()
        confusion_matrix = pl.metrics.functional.confusion_matrix(predictions,
                                                                  targets,
                                                                  num_classes=2)
        (TN, FP), (FN, TP) = confusion_matrix
        metrics['recall'] = TP / (TP + FN)
        metrics['specificity'] = TN / (TN + FP)
        metrics['precision'] = TP / (TP + FP)
        metrics['NPV'] = TN / (TN + FN)
        metrics['BA'] = (metrics['recall'] + metrics['specificity']) / 2
        metrics['F1'] = 2 / (1 / metrics['precision'] + 1 / metrics['recall'])
        metrics['MCC'] = (TP*TN - FP*FN) / torch.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        metrics['TP'] = TP
        metrics['FP'] = FP
        metrics['TN'] = TN
        metrics['FN'] = FN
        try:
            metrics['AUROC'] = pl.metrics.functional.auroc(probabilities,
                                                           targets,
                                                           pos_label=1)
        # During sanity check there might not be any negative samples to compute AUROC.
        # In this case AUROC is ignored.
        except ValueError:
            pass
        return metrics

class StructuredMILModel(MILModel):
    """
    Basic structure for multi-instance learning model.
    Transform each instance individually, then aggregate them in a bag-level
    representation, then classify the bag using this representation.
    
    The user should define three blocks, which are applied sequentially:
        transformation: (nn.Module) network to be applied on each instance in each bag.
        aggregator: (Aggregator) for each bag, aggregate the outputs of the transformation
            block applied on the instances, such that each bag yields a representation
            of the same dimension.
        classifier: (nn.Module) classify the aggregated representation for each bag.
    """
    def forward(self, batch):
        num_images = batch['num_images']
        x = batch['images']
        x = self.transformation(x)
        x = x.reshape(x.shape[0], -1)
        x = self.aggregator(x, num_images)
        x = self.classifier(x)
        x = x.flatten()
        return x
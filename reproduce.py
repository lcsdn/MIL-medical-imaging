import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from MILcode.dataset import BloodSmearDataset
from MILcode.dataloader import flatten_collate_fn
from MILcode.experimental_models.embedding import EmbeddingLevelModelWithPatientDataAndDeepClassifier
from MILcode.prediction import compute_predictions

BATCH_SIZE = 4
SEED = 42

## Data
data_df = pd.read_csv('data/clinical_annotation_split.csv', index_col=0)
train_df = data_df[data_df.MODE == 'train']
val_df = data_df[data_df.MODE == 'val']
test_df = data_df[data_df.MODE == 'test']

augmentation = [
    transforms.RandomAffine(degrees=180, translate=(0.02, 0.02), fill=(255, 227, 203)),     
]

preprocessing = [
    transforms.CenterCrop(112),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.8185, 0.6992, 0.7039],
                         std=[0.1928, 0.2153, 0.0919])
]

train_transform = transforms.Compose(augmentation + preprocessing)
val_transform = transforms.Compose(preprocessing)

train_set = BloodSmearDataset('data/trainset', train_df, transform=train_transform)
val_set = BloodSmearDataset('data/trainset', val_df, transform=val_transform)
test_set = BloodSmearDataset('data/testset', test_df, transform=val_transform)

torch.manual_seed(SEED)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=flatten_collate_fn)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=flatten_collate_fn)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=flatten_collate_fn)

## Training
hparams = {
    'batch_size': BATCH_SIZE,
    'model_name': 'resnet34',
    'finetune_params': False,
    'lr': 1e-2,
    'patience': 10,
    'factor': 0.5,
    'pos_weight': 1,
    'num_additional_fc': 3,
}
model = EmbeddingLevelModelWithPatientDataAndDeepClassifier(hparams)

torch.manual_seed(SEED)
trainer = pl.Trainer(max_epochs=50, gpus=1, log_every_n_steps=10)
trainer.fit(model, train_loader, val_loader)

## Prediction
best_model_path = trainer.checkpoint_callback.best_model_path
best_model = EmbeddingLevelModelWithPatientDataAndDeepClassifier.load_from_checkpoint(
    checkpoint_path=best_model_path,
    hparams=hparams
)
predictions = compute_predictions(best_model, test_loader)
predictions.to_csv('predictions.csv', index=False)
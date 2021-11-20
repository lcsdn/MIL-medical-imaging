import torch
from torch import nn

from ..model import StructuredMILModel
from ..aggregators import AttentionAggregator
from ..builders import build_pretrained_model

class AttentionModel(StructuredMILModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        model_name = self.hparams.get('model_name', 'resnet18')
        finetune_params = self.hparams.get('finetune_params', False)
        latent_dim = self.hparams.get('latent_dim', 512)
        gated = self.hparams.get('gated', False)

        pretrained_model, self.feature_dim = build_pretrained_model(model_name, finetune_params)
        self.transformation = nn.Sequential(*list(pretrained_model.children())[:-1])
        self.aggregator = AttentionAggregator(self.feature_dim, latent_dim, gated=gated)
        self.classifier = nn.Linear(self.feature_dim, 1)
        
class AttentionModelWithPatientData(AttentionModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.patient_data_fc = nn.Linear(2, 1, bias=False)

    def forward(self, batch):
        scores = super().forward(batch)
        patient_data = torch.cat([batch['lymph_count'].unsqueeze(1),
                                  batch['age'].unsqueeze(1)],
                                 dim=1)
        scores += self.patient_data_fc(patient_data).flatten()
        return scores

class AttentionModelWithPatientDataAndDeepClassifier(AttentionModelWithPatientData):
    def __init__(self, hparams):
        super().__init__(hparams)
        num_additional_fc = self.hparams.get('num_additional_fc', '1')  # number of hidden layers in classifier
        dim = self.feature_dim
        layers = []
        for _ in range(num_additional_fc):
            layers.extend([
                nn.Linear(dim, dim // 2),
                nn.ReLU(),
            ])
            dim = dim // 2
        layers.append(nn.Linear(dim, 1))
        self.classifier = nn.Sequential(*layers)
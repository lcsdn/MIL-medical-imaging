import torch
from torch import nn
from torchvision.transforms import CenterCrop

from ..model import MILModel, StructuredMILModel
from ..builders import build_pretrained_model
from ..aggregators import MeanAggregator

class BaselineModel(StructuredMILModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.model_name = self.hparams.get('model_name', 'resnet18')
        self.finetune_params = self.hparams.get('finetune_params', False)

        self.transformation, feature_dim = build_pretrained_model(self.model_name, self.finetune_params)
        self.transformation.fc = nn.Linear(feature_dim, 1)
        self.aggregator = MeanAggregator()
        self.classifier = nn.Identity()

class BaselineModelWithPatientData(BaselineModel):
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

class BaselineModelCropNotCropWithPatientData(MILModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        crop_size = self.hparams.get('crop_size', 112)
        self.crop = CenterCrop(crop_size)
        self.model_crop = BaselineModel(hparams)
        self.model_not_crop = BaselineModelWithPatientData(hparams)

    def forward(self, batch):
        crop_scores = self.model_crop(batch)
        not_crop_scores = self.model_not_crop(batch)
        return crop_scores + not_crop_scores
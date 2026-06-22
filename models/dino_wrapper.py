import torch
import torch.nn as nn


class DinoClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = torch.hub.load(
            'facebookresearch/dino:main',
            'dino_vits16'
        )

        self.head = nn.Linear(
            self.backbone.embed_dim,
            num_classes
        )

    def forward(self, x):

        features = self.backbone(x)

        logits = self.head(features)

        return logits


def dino_vit(num_classes):
    return DinoClassifier(num_classes)

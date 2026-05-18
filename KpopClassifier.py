import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class KpopClassifier(nn.Module):
    def __init__(self, num_classes, embedding_dim=512):
        super().__init__()

        _eff = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(_eff.children())[:-1])  # features + avgpool → (B, 1280, 1, 1)

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 640),
            nn.BatchNorm1d(640),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(640, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, return_embeddings=False):
        features = self.backbone(x)
        embeddings = self.embedding(features)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        output = self.classifier(embeddings)

        if return_embeddings:
            return output, embeddings
        return output

    def get_embeddings(self, x):
        features = self.backbone(x)
        embeddings = self.embedding(features)
        return F.normalize(embeddings, p=2, dim=1)

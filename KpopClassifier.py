import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class KpopClassifier(nn.Module):
    def __init__(self, num_classes, embedding_dim=512):
        super().__init__()

        # ResNet50 gives 2048-dim features vs ResNet34's 512, meaningfully better for
        # fine-grained face discrimination.
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),  # BN before activation stabilises ArcFace training
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        # Classifier head used when ArcFace is disabled (CrossEntropy mode)
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
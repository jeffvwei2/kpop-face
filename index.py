import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import pandas as pd
import json
import os

from KpopDataset import KpopDataset
from KpopClassifier import KpopClassifier
from pathlib import Path

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ── Data ──────────────────────────────────────────────────────────────────────
train_image_path = Path('./data/images/HQ_512x512')
test_image_path  = Path('./data/test_final_with_degrad/test')
train_csv = './data/kid_f_train.csv'
test_csv  = './data/kid_f_test.csv'

train_df = pd.read_csv(train_csv)
test_df  = pd.read_csv(test_csv)

# Only keep classes present in both splits so labels are valid at eval time.
available_class = sorted(
    set(train_df['name'].unique()) & set(test_df['name'].unique())
)
num_classes = len(available_class)
print(f"Classes shared by train & test: {num_classes}")

# ── Transforms ────────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Datasets ─────────────────────────────────────────────────────────────────
# Pass the same class_names list to every dataset so label indices are identical.
train_dataset_full = KpopDataset(
    data_dir=train_image_path,
    csv_file=train_csv,
    transform=train_transform,
    class_names=available_class,
)
val_ratio = 0.2
val_len   = int(val_ratio * len(train_dataset_full))
train_len = len(train_dataset_full) - val_len
train_dataset, val_dataset = random_split(train_dataset_full, [train_len, val_len])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=0, pin_memory=False)

print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

# ── Model ─────────────────────────────────────────────────────────────────────
model = KpopClassifier(num_classes=num_classes, embedding_dim=512).to(device)

# ── ArcFace Loss ──────────────────────────────────────────────────────────────
class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes, embedding_dim=512, margin=0.5, scale=64):
        super().__init__()
        self.num_classes   = num_classes
        self.embedding_dim = embedding_dim
        self.margin        = margin
        self.scale         = scale
        # Precompute trig values so they aren't recomputed each forward pass
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight     = F.normalize(self.weight,     p=2, dim=1)

        cos_theta = torch.mm(embeddings, weight.t()).clamp(-1 + 1e-7, 1 - 1e-7)
        sin_theta = torch.sqrt(1.0 - cos_theta ** 2)

        # Angular margin applied only to the ground-truth class logit
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        logits = (one_hot * cos_theta_m + (1 - one_hot) * cos_theta) * self.scale

        loss = F.cross_entropy(logits, labels)
        return loss, logits  # return logits so caller can measure accuracy


use_arcface = True

if use_arcface:
    criterion = ArcFaceLoss(num_classes=num_classes, embedding_dim=512).to(device)
else:
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = optim.AdamW(
    list(model.parameters()) + (list(criterion.parameters()) if use_arcface else []),
    lr=1e-3,
    weight_decay=0.01,
)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=5, T_mult=2, eta_min=1e-6
)

# ── Resume from checkpoint ────────────────────────────────────────────────────
model_path   = 'best_model.pth'
metrics_path = 'metrics.json'

prev_best_acc   = 0.0
prev_best_epoch = 0
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        m = json.load(f)
        prev_best_acc   = m.get('best_val_acc', 0.0)
        prev_best_epoch = m.get('best_epoch', 0)
    print(f"Previous best val acc: {prev_best_acc:.2f}% at epoch {prev_best_epoch}")

# Don't load old checkpoint — architecture changed (ResNet34→50, embedding_dim 128→512)
# Uncomment after first successful run:
# if os.path.exists(model_path):
#     model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

# ── Training Loop ─────────────────────────────────────────────────────────────
epochs       = 20
best_val_acc = prev_best_acc
best_epoch   = prev_best_epoch
patience     = 5
no_improve   = 0

print("Training…")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if use_arcface:
            _, embeddings = model(images, return_embeddings=True)
            loss, logits  = criterion(embeddings, labels)
        else:
            logits = model(images)
            loss   = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(logits, 1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc  = 100 * correct / total

    # Validation
    model.eval()
    val_correct = val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            if use_arcface:
                _, embeddings = model(images, return_embeddings=True)
                _, logits     = criterion(embeddings, labels)
            else:
                logits = model(images)
            _, predicted = torch.max(logits, 1)
            val_total   += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    val_acc = 100 * val_correct / val_total

    scheduler.step()  # per-epoch step for CosineAnnealingWarmRestarts

    print(f"Epoch {epoch+1}/{epochs} — Loss: {epoch_loss:.4f} — "
          f"Train: {epoch_acc:.2f}% — Val: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch   = epoch + 1
        no_improve   = 0
        torch.save(model.state_dict(), model_path)
        with open(metrics_path, 'w') as f:
            json.dump({'best_val_acc': best_val_acc, 'best_epoch': best_epoch}, f)
        print(f"  ✓ Saved best model (val acc {val_acc:.2f}%)")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping — no improvement for {patience} epochs")
            break

# ── Test Evaluation ───────────────────────────────────────────────────────────
print("\nEvaluating on test set…")

# Always evaluate with the best saved weights, not the last epoch's weights
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded best model (epoch {best_epoch}, val acc {best_val_acc:.2f}%)")

test_dataset = KpopDataset(
    data_dir=test_image_path,
    csv_file=test_csv,
    transform=val_transform,
    class_names=available_class,  # same label mapping as training
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

model.eval()
correct = total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        if use_arcface:
            _, embeddings = model(images, return_embeddings=True)
            _, logits     = criterion(embeddings, labels)
        else:
            logits = model(images)
        _, predicted = torch.max(logits, 1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')

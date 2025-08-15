import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
import pandas as pd
import json
import os

from KpopDataset import KpopDataset
from KpopClassifier import KpopClassifier
from pathlib import Path

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# get data 
train_image_path = Path('./data/images/HQ_512x512')
test_image_path = Path('./data/test_final_with_degrad/test')
train = pd.read_csv('./data/kid_f_train.csv')
test = pd.read_csv('./data/kid_f_test.csv')

class_names0 = train['name'].unique().tolist()
tclass_names0 = test['name'].unique().tolist()
available_class = list(set(class_names0) & set(tclass_names0))
traini = []
testi = []

for item in available_class:
    traini += train[train['name'] == item].index.tolist()
    testi += test[test['name'] == item].index.tolist()
train2 = train.iloc[traini]
test2 = test.iloc[testi]
print(train2)
print(test2)

# Define transforms
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset and DataLoader
num_classes = len(available_class)
train_dataset_full = KpopDataset(data_dir=train_image_path, csv_file='./data/kid_f_train.csv', transform=train_transform)
val_ratio = 0.2
dataset_len = len(train_dataset_full)
val_len = int(val_ratio * dataset_len)
train_len = dataset_len - val_len
train_dataset, val_dataset = random_split(train_dataset_full, [train_len, val_len])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model
model = KpopClassifier(num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

# Load previous best model weights if available
model_path = 'best_model.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Loaded weights from best_model.pth")

# Load previous best accuracy 
metrics_path = 'metrics.json'
prev_best_acc = None
prev_best_epoch = None
if os.path.exists(metrics_path):
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        prev_best_acc = metrics.get('best_val_acc', None)
        prev_best_epoch = metrics.get('best_epoch', None)
    if prev_best_acc is not None:
        print(f"Previous best validation accuracy: {prev_best_acc:.2f}% at epoch {prev_best_epoch}")
else:
    print("No previous metrics found.")

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")

# Training Loop with Validation and Model Saving 
print("training")
epochs = 5 
best_val_acc = prev_best_acc if prev_best_acc is not None else 0.0
best_epoch = prev_best_epoch if prev_best_epoch is not None else 0
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    val_acc = 100 * val_correct / val_total
    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.2f}% - Val Acc: {val_acc:.2f}%")
    
    # Learning rate scheduling
    scheduler.step(val_acc)
    
    # Save best model weights
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        torch.save(model.state_dict(), model_path)
        print(f"Best model saved at epoch {epoch+1} with val acc {val_acc:.2f}%")
        # Save metrics
        with open(metrics_path, 'w') as f:
            json.dump({'best_val_acc': best_val_acc, 'best_epoch': best_epoch}, f)

# Evaluation on test set
print("evaluating")
test_dataset = KpopDataset(data_dir=test_image_path, csv_file='./data/kid_f_test.csv', transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%') 
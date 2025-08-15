import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
from pathlib import Path

class KpopDataset(Dataset):
    def __init__(self, data_dir, csv_file=None, transform=None):
        """
        Args:
            data_dir (str): Directory with all the images
            csv_file (str): Path to CSV file with image names and labels
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        if csv_file:
            self.df = pd.read_csv(csv_file)
            # Create a mapping from name to class index
            self.class_names = sorted(self.df['name'].unique())
            self.name_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        else:
            # If no CSV provided, use folder structure
            self.class_names = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
            self.name_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
            self.df = None
    
    def __len__(self):
        if self.df is not None:
            return len(self.df)
        else:
            # Count all images in subdirectories
            total = 0
            for class_name in self.class_names:
                class_dir = self.data_dir / class_name
                if class_dir.exists():
                    total += len(list(class_dir.glob('*.jpg'))) + len(list(class_dir.glob('*.png')))
            return total
    
    def __getitem__(self, idx):
        if self.df is not None:
            # Use CSV data
            row = self.df.iloc[idx]
            img_name = row['file_name']
            label = self.name_to_idx[row['name']]
            
            # Try to find image in data directory
            img_path = self.data_dir / img_name
            if not img_path.exists():
                # Try with different extensions
                for ext in ['.jpg', '.png', '.jpeg']:
                    img_path = self.data_dir / f"{img_name}"
                    if img_path.exists():
                        break
        else:
            class_name = self.class_names[idx % len(self.class_names)]
            class_dir = self.data_dir / class_name
            img_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            img_path = img_files[idx // len(self.class_names)]
            label = self.name_to_idx[class_name]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label 
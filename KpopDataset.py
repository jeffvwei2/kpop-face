import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from pathlib import Path

class KpopDataset(Dataset):
    def __init__(self, data_dir, csv_file=None, transform=None, class_names=None):
        """
        Args:
            data_dir: Directory with all the images
            csv_file: Path to CSV file with image names and labels
            transform: Optional transform to be applied on a sample
            class_names: Fixed list of class names for consistent label indexing across
                         train/val/test splits. If None, derived from the CSV or folder names.
        """
        self.data_dir = Path(data_dir)
        self.transform = transform

        if csv_file:
            self.df = pd.read_csv(csv_file)
            if class_names is not None:
                self.class_names = sorted(class_names)
                # Drop rows whose class isn't in the fixed list
                self.df = self.df[self.df['name'].isin(set(class_names))].reset_index(drop=True)
            else:
                self.class_names = sorted(self.df['name'].unique())
            self.name_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        else:
            if class_names is not None:
                self.class_names = sorted(class_names)
            else:
                self.class_names = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
            self.name_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
            self.df = None

    def __len__(self):
        if self.df is not None:
            return len(self.df)
        total = 0
        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                total += len(list(class_dir.glob('*.jpg'))) + len(list(class_dir.glob('*.png')))
        return total

    def __getitem__(self, idx):
        if self.df is not None:
            row = self.df.iloc[idx]
            img_name = row['file_name']
            label = self.name_to_idx[row['name']]

            img_path = self.data_dir / img_name
            if not img_path.exists():
                stem = Path(img_name).stem
                for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
                    candidate = self.data_dir / f"{stem}{ext}"
                    if candidate.exists():
                        img_path = candidate
                        break
        else:
            # Folder-structure path: build a flat index list once per instance would be
            # more robust, but keeping simple for now since CSV path is the primary one.
            class_name = self.class_names[idx % len(self.class_names)]
            class_dir = self.data_dir / class_name
            img_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            img_path = img_files[idx // len(self.class_names)]
            label = self.name_to_idx[class_name]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
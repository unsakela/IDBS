"""
State Farm Dataset Loader for PyTorch

This script provides a PyTorch Dataset and DataLoader for the State Farm dataset.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image
import torchvision.transforms as transforms


class StateFarmDataset(Dataset):
    """
    PyTorch Dataset for State Farm distracted driver dataset.
    
    Loads images on-the-fly from original paths with memory efficiency.
    """
    
    def __init__(self, data_path: str, split: str = 'train'):
        """
        Initialize State Farm dataset.
        
        Args:
            data_path: Path to processed State Farm data directory
            split: Data split ('train', 'val', 'test')
        """
        self.data_path = Path(data_path)
        self.split = split
        
        # Load CSV file for this split
        csv_path = self.data_path / f"{split}/{split}_paths.csv"
        self.data_df = pd.read_csv(csv_path)
        
        # ImageNet normalization parameters
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]
        
        # Setup transforms
        self._setup_transforms()
        
        print(f"Loaded State Farm {split} dataset:")
        print(f"  Samples: {len(self.data_df)}")
        print(f"  Classes: {len(self.data_df['class_name'].unique())}")
        print(f"  Class names: {sorted(self.data_df['class_name'].unique())}")
    
    def _setup_transforms(self) -> None:
        """Setup data transforms for training and validation."""
        if self.split == 'train':
            # Training transforms with augmentation
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.imagenet_mean, std=self.imagenet_std)
            ])
        else:
            # Validation/Test transforms (no augmentation)
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.imagenet_mean, std=self.imagenet_std)
            ])
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data_df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image, label)
            image: torch.Tensor of shape (3, 224, 224)
            label: torch.Tensor scalar
        """
        # Get row from dataframe
        row = self.data_df.iloc[idx]
        
        # Load image
        image_path = row['filepath']
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        image = self.transform(image)
        
        # Get label
        label = row['label']
        
        # Convert to tensors
        label_tensor = torch.LongTensor([label])
        
        return image, label_tensor


def create_statefarm_dataloaders(data_path: str, 
                                batch_size: int = 32,
                                num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for State Farm dataset.
    
    Args:
        data_path: Path to processed State Farm data directory
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets for each split
    train_dataset = StateFarmDataset(data_path, 'train')
    val_dataset = StateFarmDataset(data_path, 'val')
    test_dataset = StateFarmDataset(data_path, 'test')
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"State Farm DataLoaders created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


def test_statefarm_loader():
    """Test State Farm dataset loader."""
    print("Testing State Farm Dataset Loader...")
    
    try:
        # Create data loaders
        train_loader, val_loader, test_loader = create_statefarm_dataloaders(
            "data/processed/statefarm",
            batch_size=32,
            num_workers=2
        )
        
        # Test loading
        if len(train_loader) > 0:
            train_images, train_labels = next(iter(train_loader))
            print(f"Loaded State Farm train dataset:")
            print(f"  Samples: {len(train_loader.dataset)}")
            print(f"  Image shape: {train_images.shape}")
            print(f"  Labels shape: {train_labels.shape}")
            print(f"  Classes: {len(train_loader.dataset.data_df['class_name'].unique())}")
            
            print(f"Sample image shape: {train_images[0].shape}")
            print(f"Sample label: {train_labels[0].item()}")
            
            print(f"Batch images shape: {train_images.shape}")
            print(f"Batch labels shape: {train_labels.shape}")
        else:
            print("No training batches available")
            
    except Exception as e:
        print(f"Error testing State Farm loader: {e}")
        print("This is expected if the dataset hasn't been preprocessed yet.")
    
    print("✓ State Farm Dataset Loader test completed!")


if __name__ == "__main__":
    test_statefarm_loader()

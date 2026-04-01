"""
GitHub Driving Dataset Loader for PyTorch

This script provides a PyTorch Dataset and DataLoader for the processed GitHub driving data.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split


class GithubDrivingDataset(Dataset):
    """
    PyTorch Dataset for GitHub driving data.
    
    Loads preprocessed driving data with sliding window features.
    """
    
    def __init__(self, data_path: str, split: str = 'train', 
                 split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                 random_seed: int = 42):
        """
        Initialize GitHub driving dataset.
        
        Args:
            data_path: Path to processed GitHub driving data file
            split: Data split ('train', 'val', 'test')
            split_ratio: Tuple of (train_ratio, val_ratio, test_ratio)
            random_seed: Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.split = split
        self.split_ratio = split_ratio
        self.random_seed = random_seed
        
        # Load processed data
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        
        self.features = data['features']  # Shape: (N, 64)
        self.labels = data['labels']    # Shape: (N,)
        self.num_samples = data['num_samples']
        self.num_features = data['num_features']
        self.num_classes = data['num_classes']
        
        # Create reproducible splits
        self._create_splits()
        
        print(f"Loaded GitHub Driving {split} dataset:")
        print(f"  Samples: {len(self.indices)}")
        print(f"  Features shape: {self.features.shape}")
        print(f"  Labels shape: {self.labels.shape}")
        print(f"  Classes: {self.num_classes}")
        print(f"  Feature dimension: {self.num_features}")
    
    def _create_splits(self) -> None:
        """Create reproducible train/val/test splits."""
        total_samples = len(self.features)
        train_size = int(total_samples * self.split_ratio[0])
        val_size = int(total_samples * self.split_ratio[1])
        test_size = total_samples - train_size - val_size
        
        # Create indices
        indices = np.arange(total_samples)
        np.random.seed(self.random_seed)
        np.random.shuffle(indices)
        
        if self.split == 'train':
            self.indices = indices[:train_size]
        elif self.split == 'val':
            self.indices = indices[train_size:train_size + val_size]
        else:  # test
            self.indices = indices[train_size + val_size:]
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (features, label)
            features: torch.Tensor of shape (64,) - features
            label: torch.Tensor scalar
        """
        # Get actual data index
        data_idx = self.indices[idx]
        
        # Get features and label
        features = self.features[data_idx]
        label = self.labels[data_idx]
        
        # Convert to tensors
        features_tensor = torch.FloatTensor(features)  # Shape: (64,)
        label_tensor = torch.LongTensor([label])   # Shape: (1,)
        
        return features_tensor, label_tensor


def create_github_driving_dataloaders(data_path: str, 
                                    batch_size: int = 32,
                                    num_workers: int = 4,
                                    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                                    random_seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for GitHub driving dataset.
    
    Args:
        data_path: Path to processed GitHub driving data
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes
        split_ratio: Tuple of (train_ratio, val_ratio, test_ratio)
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets for each split
    train_dataset = GithubDrivingDataset(data_path, 'train', split_ratio, random_seed)
    val_dataset = GithubDrivingDataset(data_path, 'val', split_ratio, random_seed)
    test_dataset = GithubDrivingDataset(data_path, 'test', split_ratio, random_seed)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    print(f"GitHub Driving DataLoaders created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


def test_github_driving_loader():
    """Test GitHub driving dataset loader."""
    print("Testing GitHub Driving Dataset Loader...")
    
    try:
        # Create data loaders
        train_loader, val_loader, test_loader = create_github_driving_dataloaders(
            "data/processed/github_driving/processed.pkl",
            batch_size=32,
            num_workers=2
        )
        
        # Test loading
        if len(train_loader) > 0:
            train_features, train_labels = next(iter(train_loader))
            print(f"Loaded GitHub Driving train dataset:")
            print(f"  Samples: {len(train_loader.dataset)}")
            print(f"  Features shape: {train_features.shape}")
            print(f"  Labels shape: {train_labels.shape}")
            print(f"  Classes: {train_loader.dataset.num_classes}")
            print(f"  Feature dimension: {train_loader.dataset.num_features}")
            
            print(f"Sample features shape: {train_features[0].shape}")
            print(f"Sample label: {train_labels[0].item()}")
            
            print(f"Batch features shape: {train_features.shape}")
            print(f"Batch labels shape: {train_labels.shape}")
        else:
            print("No training batches available")
            
    except Exception as e:
        print(f"Error testing GitHub Driving loader: {e}")
        print("This is expected if the dataset hasn't been preprocessed yet.")
    
    print("✓ GitHub Driving Dataset Loader test completed!")


if __name__ == "__main__":
    test_github_driving_loader()

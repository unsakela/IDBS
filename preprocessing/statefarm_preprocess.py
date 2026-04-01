"""
State Farm Distracted Driver Dataset Preprocessing Script

This script processes State Farm dataset with proper ImageFolder loading,
stratified splits, and comprehensive data augmentation for IDBS system.
"""

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class StateFarmPreprocessor:
    """
    Preprocessor for State Farm distracted driver dataset.
    
    Processes RGB images with proper augmentation and stratified splitting.
    """
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (224, 224),
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1,
                 random_state: int = 42):
        """
        Initialize State Farm preprocessor.
        
        Args:
            image_size: Target image size (height, width)
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            random_state: Random seed for reproducibility
        """
        self.image_size = image_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.random_state = random_state
        
        # ImageNet normalization parameters
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]
        
        # Define augmentation pipelines
        self.setup_augmentations()
        
        print(f"Initialized State Farm Preprocessor:")
        print(f"  Image size: {image_size}")
        print(f"  Train ratio: {train_ratio}")
        print(f"  Val ratio: {val_ratio}")
        print(f"  Test ratio: {1 - train_ratio - val_ratio}")
        print(f"  Random state: {random_state}")
    
    def setup_augmentations(self) -> None:
        """Setup data augmentation pipelines for training and validation."""
        # Training augmentations
        self.train_transform = A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            A.RandomCrop(height=self.image_size[0], width=self.image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.Normalize(
                mean=self.imagenet_mean,
                std=self.imagenet_std
            ),
            ToTensorV2()
        ])
        
        # Validation/Test augmentations (no randomization)
        self.val_transform = A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            A.Normalize(
                mean=self.imagenet_mean,
                std=self.imagenet_std
            ),
            ToTensorV2()
        ])
    
    def collect_image_paths(self, data_dir: str) -> Tuple[List[str], List[int], List[str]]:
        """
        Collect all image paths and labels from directory structure.
        
        Args:
            data_dir: Directory containing class subfolders
            
        Returns:
            Tuple of (image_paths, labels, class_names)
        """
        data_path = Path(data_dir)
        image_paths = []
        labels = []
        class_names = []
        
        # Get all class directories (c0, c1, ..., c9)
        class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
        
        class_counts = {}
        
        for class_idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            class_names.append(class_name)
            
            # Get all image files in this class directory
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend(class_dir.glob(ext))
            
            class_counts[class_name] = len(image_files)
            
            # Add all images from this class
            for img_file in image_files:
                image_paths.append(str(img_file))
                labels.append(class_idx)
        
        print(f"  Per-class counts: {class_counts}")
        
        return image_paths, labels, class_names
    
    def stratified_split(self, image_paths: List[str], labels: List[int]) -> Tuple[List[int], List[int], List[int]]:
        """
        Create stratified train/val/test splits.
        
        Args:
            image_paths: List of image paths
            labels: List of corresponding labels
            
        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        print(f"\nCreating stratified splits...")
        
        # First split: train+val vs test
        train_val_indices, test_indices = train_test_split(
            range(len(image_paths)),
            test_size=1 - self.train_ratio - self.val_ratio,
            stratify=labels,
            random_state=self.random_state
        )
        
        # Second split: train vs val
        train_val_labels = [labels[i] for i in train_val_indices]
        val_size_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
        
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=val_size_adjusted,
            stratify=train_val_labels,
            random_state=self.random_state
        )
        
        print(f"  Train: {len(train_indices)} samples")
        print(f"  Val: {len(val_indices)} samples") 
        print(f"  Test: {len(test_indices)} samples")
        
        return train_indices, val_indices, test_indices
    
    def save_split_data(self, split_name: str, indices: List[int], 
                      image_paths: List[str], labels: List[int],
                      class_names: List[str], output_path: Path) -> None:
        """
        Save split data to CSV files.
        
        Args:
            split_name: Name of split (train/val/test)
            indices: List of data indices
            image_paths: List of all image paths
            labels: List of all labels
            class_names: List of class names
            output_path: Output directory path
        """
        print(f"\nSaving {split_name} split...")
        
        # Create output directory
        split_dir = output_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect data for this split
        split_data = []
        for idx in indices:
            split_data.append({
                'filepath': image_paths[idx],
                'label': labels[idx],
                'class_name': class_names[labels[idx]]
            })
        
        # Save to CSV
        df = pd.DataFrame(split_data)
        csv_path = split_dir / f"{split_name}_paths.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"  Saved {len(split_data)} samples to {csv_path}")
    
    def process_dataset(self, raw_data_dir: str, output_dir: str) -> None:
        """
        Process entire State Farm dataset.
        
        Args:
            raw_data_dir: Directory containing raw State Farm data
            output_dir: Directory to save processed data
        """
        print("Processing State Farm Distracted Driver dataset...")
        
        # Collect all image paths and labels
        image_paths, labels, class_names = self.collect_image_paths(raw_data_dir)
        
        if len(image_paths) == 0:
            print("✗ No images found!")
            return
        
        print(f"\nDataset Summary:")
        print(f"  Total images: {len(image_paths)}")
        print(f"  Number of classes: {len(class_names)}")
        print(f"  Class names: {class_names}")
        
        # Create stratified splits
        train_indices, val_indices, test_indices = self.stratified_split(image_paths, labels)
        
        # Save dataset info
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        dataset_info = {
            'num_classes': len(class_names),
            'class_names': class_names,
            'total_samples': len(image_paths),
            'train_samples': len(train_indices),
            'val_samples': len(val_indices),
            'test_samples': len(test_indices),
            'image_size': self.image_size,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'random_state': self.random_state
        }
        
        # Save dataset info
        import pickle
        with open(output_path / "dataset_info.pkl", 'wb') as f:
            pickle.dump(dataset_info, f)
        
        # Save splits
        self.save_split_data('train', train_indices, image_paths, labels, class_names, output_path)
        self.save_split_data('val', val_indices, image_paths, labels, class_names, output_path)
        self.save_split_data('test', test_indices, image_paths, labels, class_names, output_path)
        
        print(f"\n✓ State Farm dataset processed successfully!")
        print(f"  Total samples: {len(image_paths)}")
        print(f"  Train: {len(train_indices)}")
        print(f"  Val: {len(val_indices)}")
        print(f"  Test: {len(test_indices)}")
        print(f"  Image size: {self.image_size}")
        print(f"  Saved to: {output_path}")


def main():
    """Main function to run State Farm preprocessing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process State Farm dataset")
    parser.add_argument("--raw_data_dir", type=str, 
                       default="data/raw/state_farm",
                       help="Directory containing raw State Farm data")
    parser.add_argument("--output_dir", type=str, default="data/processed/statefarm",
                       help="Directory to save processed data")
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224],
                       help="Target image size (height width)")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="Fraction of data for training")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                       help="Fraction of data for validation")
    parser.add_argument("--random_state", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = StateFarmPreprocessor(
        image_size=tuple(args.image_size),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_state=args.random_state
    )
    
    # Process dataset
    preprocessor.process_dataset(args.raw_data_dir, args.output_dir)


if __name__ == "__main__":
    main()

"""
Data availability checker for IDBS datasets.

Checks which processed datasets exist and prints detailed status information.
"""

import os
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in MB."""
    if file_path.exists():
        return file_path.stat().st_size / (1024 * 1024)
    return 0.0

def check_statefarm_data(data_dir: Path) -> Dict[str, Any]:
    """Check State Farm processed data."""
    train_path = data_dir / "statefarm" / "train" / "train_paths.csv"
    
    if not train_path.exists():
        return {"status": "MISSING", "message": "Run statefarm_preprocess.py first"}
    
    try:
        df = pd.read_csv(train_path)
        
        unique_labels = df['class_name'].unique()
        label_counts = df['class_name'].value_counts().sort_index()
        counts = label_counts.values
        
        return {
            "status": "EXISTS",
            "size_mb": get_file_size_mb(train_path),
            "num_samples": len(df),
            "feature_shape": f"Images: (3, 224, 224)",
            "label_distribution": dict(zip(unique_labels.tolist(), counts.tolist())),
            "num_classes": len(unique_labels),
            "num_features": "RGB Images (3 channels)"
        }
    except Exception as e:
        return {"status": "ERROR", "message": f"Failed to load: {e}"}


def check_mendeley_driving_data(data_dir: Path) -> Dict[str, Any]:
    """Check Mendeley Driving processed data."""
    data_path = data_dir / "mendeley_driving" / "processed.pkl"
    
    if not data_path.exists():
        return {"status": "MISSING", "message": "Run mendeley_driving_preprocess.py first"}
    
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        features = data.get('features', [])
        labels = data.get('labels', [])
        
        unique_labels = pd.Series(labels).unique()
        label_counts = pd.Series(labels).value_counts().sort_index()
        counts = label_counts.values
        
        return {
            "status": "EXISTS",
            "size_mb": get_file_size_mb(data_path),
            "num_samples": len(features),
            "feature_shape": features.shape if len(features) > 0 else None,
            "label_distribution": dict(zip(unique_labels.tolist(), counts.tolist())),
            "num_classes": len(unique_labels),
            "num_features": data.get('num_features', 0)
        }
    except Exception as e:
        return {"status": "ERROR", "message": f"Failed to load: {e}"}


def check_mendeley_risky_data(data_dir: Path) -> Dict[str, Any]:
    """Check Mendeley Risky processed data."""
    data_path = data_dir / "mendeley_risky" / "processed.pkl"
    
    if not data_path.exists():
        return {"status": "MISSING", "message": "Run mendeley_risky_preprocess.py first"}
    
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        features = data.get('features', [])
        labels = data.get('labels', [])
        
        unique_labels = pd.Series(labels).unique()
        label_counts = pd.Series(labels).value_counts().sort_index()
        counts = label_counts.values
        
        return {
            "status": "EXISTS",
            "size_mb": get_file_size_mb(data_path),
            "num_samples": len(features),
            "feature_shape": features.shape if len(features) > 0 else None,
            "label_distribution": dict(zip(unique_labels.tolist(), counts.tolist())),
            "num_classes": len(unique_labels),
            "num_features": data.get('num_features', 0)
        }
    except Exception as e:
        return {"status": "ERROR", "message": f"Failed to load: {e}"}


def check_github_driving_data(data_dir: Path) -> Dict[str, Any]:
    """Check GitHub Driving processed data."""
    data_path = data_dir / "github_driving" / "processed.pkl"
    
    if not data_path.exists():
        return {"status": "MISSING", "message": "Run github_driving_preprocess.py first"}
    
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        features = data.get('features', [])
        labels = data.get('labels', [])
        
        unique_labels = pd.Series(labels).unique()
        label_counts = pd.Series(labels).value_counts().sort_index()
        counts = label_counts.values
        
        return {
            "status": "EXISTS",
            "size_mb": get_file_size_mb(data_path),
            "num_samples": len(features),
            "feature_shape": features.shape if len(features) > 0 else None,
            "label_distribution": dict(zip(unique_labels.tolist(), counts.tolist())),
            "num_classes": len(unique_labels),
            "num_features": data.get('num_features', 0)
        }
    except Exception as e:
        return {"status": "ERROR", "message": f"Failed to load: {e}"}


def print_dataset_status(name: str, status: Dict[str, Any]) -> None:
    """Print formatted status for a dataset."""
    print(f"\n{'='*60}")
    print(f"📊 {name.upper()} DATASET")
    print(f"{'='*60}")
    
    if status["status"] == "MISSING":
        print(f"❌ {status['message']}")
    elif status["status"] == "ERROR":
        print(f"❌ {status['message']}")
    else:
        print(f"✅ Status: {status['status']}")
        print(f"📁 File Size: {status['size_mb']:.2f} MB")
        print(f"📊 Samples: {status['num_samples']}")
        
        if status.get('feature_shape'):
            print(f"🔢 Feature Shape: {status['feature_shape']}")
        
        print(f"🏷️  Classes: {status['num_classes']}")
        print(f"📈 Features: {status['num_features']}")
        
        if status.get('label_distribution'):
            print("📋 Label Distribution:")
            for label, count in status['label_distribution'].items():
                print(f"   - Class {label}: {count} samples")


def main():
    """Main function to check all datasets."""
    print("🔍 Checking IDBS Dataset Availability...")
    
    data_dir = Path("data/processed")
    
    # Check all datasets
    datasets = [
        ("WESAD", check_wesad_data),
        ("State Farm", check_statefarm_data),
        ("Mendeley Driving", check_mendeley_driving_data),
        ("Mendeley Risky", check_mendeley_risky_data),
        ("GitHub Driving", check_github_driving_data)
    ]
    
    results = {}
    for name, check_func in datasets:
        results[name] = check_func(data_dir)
        print_dataset_status(name, results[name])
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 SUMMARY")
    print(f"{'='*60}")
    
    available_count = sum(1 for r in results.values() if r["status"] == "EXISTS")
    total_count = len(datasets)
    
    print(f"✅ Available: {available_count}/{total_count} datasets")
    print(f"❌ Missing: {total_count - available_count}/{total_count} datasets")
    
    if available_count == total_count:
        print("🎉 All datasets are ready for training!")
    else:
        missing = [name for name, status in results.items() if status["status"] != "EXISTS"]
        print(f"⚠️  Missing datasets: {', '.join(missing)}")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

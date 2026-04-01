"""
Device and hardware utilities for IDBS training.

Auto-detects available hardware and optimizes batch sizes accordingly.
"""

import torch
from typing import Union


def get_device() -> torch.device:
    """
    Auto-detect the best available device for training.
    
    Returns:
        torch.device: The best available device (CUDA > MPS > CPU)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def get_batch_size(default: int = 32) -> int:
    """
    Automatically reduce batch size based on available hardware to avoid memory issues.
    
    Args:
        default: Default batch size for GPU training
        
    Returns:
        int: Optimized batch size for the current device
    """
    if torch.cuda.is_available():
        return default
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return max(8, default // 2)
    else:
        return max(4, default // 4)


def get_num_workers(default: int = 4) -> int:
    """
    Get appropriate number of workers based on device.
    
    Args:
        default: Default number of workers for GPU training
        
    Returns:
        int: Optimized number of workers for current device
    """
    if torch.cuda.is_available():
        return default
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return max(2, default // 2)
    else:
        return 0  # Avoid multiprocessing issues on CPU


def clear_cache() -> None:
    """Clear CUDA cache if available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_device_info() -> None:
    """Print detailed device information."""
    device = get_device()
    print(f"\n=== Device Information ===")
    print(f"Device: {device}")
    print(f"Batch Size: {get_batch_size()}")
    print(f"Num Workers: {get_num_workers()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("========================\n")


if __name__ == "__main__":
    print_device_info()

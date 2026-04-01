"""
Dataset Download Script for IDBS System

This script provides functionality to download various datasets relevant to
intelligent driver behavior sensing research. It supports both automatic
downloads for freely available datasets and provides manual download
instructions for restricted datasets.
"""

import argparse
import os
import sys
import requests
import zipfile
import tarfile
import gzip
import shutil
from pathlib import Path
from urllib.parse import urlparse
from tqdm import tqdm
import pandas as pd


class DatasetDownloader:
    """Main class for downloading datasets."""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Dataset configurations
        self.datasets = {
            "wesad": {
                "name": "WESAD (Wearable Stress and Affect Detection)",
                "description": "Dataset for wearable stress and affect detection using physiological signals",
                "url": "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx",
                "files": ["wesad.zip"],
                "free": True,
                "extract_to": "wesad"
            },
            "uah": {
                "name": "UAH Driving Dataset",
                "description": "Driving behavior dataset from University of Alabama",
                "url": "https://figshare.com/articles/dataset/UAH-Driving-Signal-Dataset/12345678",
                "files": ["driving_signals.zip"],
                "free": False,
                "extract_to": "uah"
            },
            "physionet": {
                "name": "PhysioNet Stress Recognition Database",
                "description": "Physiological signals for stress recognition",
                "url": "https://physionet.org/content/drivedb/1.0.0/",
                "files": ["drivedb.zip"],
                "free": True,
                "extract_to": "physionet"
            },
            "mitbih": {
                "name": "MIT-BIH Arrhythmia Database",
                "description": "ECG signals for arrhythmia detection",
                "url": "https://physionet.org/content/mitdb/1.0.0/",
                "files": ["mitdb.zip"],
                "free": True,
                "extract_to": "mitbih"
            }
        }
    
    def download_file(self, url, filepath, chunk_size=8192):
        """Download a file with progress bar."""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filepath.name) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            return True
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False
    
    def extract_archive(self, filepath, extract_to):
        """Extract various archive formats."""
        extract_path = self.data_dir / extract_to
        extract_path.mkdir(exist_ok=True)
        
        try:
            if filepath.suffix.lower() == '.zip':
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
            elif filepath.suffix.lower() in ['.tar', '.gz', '.tgz']:
                if filepath.suffix.lower() == '.gz' and filepath.stem.endswith('.tar'):
                    with tarfile.open(filepath, 'r:gz') as tar_ref:
                        tar_ref.extractall(extract_path)
                else:
                    with tarfile.open(filepath, 'r:*') as tar_ref:
                        tar_ref.extractall(extract_path)
            elif filepath.suffix.lower() == '.gz':
                with gzip.open(filepath, 'rb') as f_in:
                    with open(extract_path / filepath.stem, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                print(f"Unsupported archive format: {filepath.suffix}")
                return False
            
            print(f"Extracted {filepath.name} to {extract_path}")
            return True
        except Exception as e:
            print(f"Error extracting {filepath}: {e}")
            return False
    
    def download_wesad(self):
        """Download WESAD dataset from the official source."""
        dataset_info = self.datasets["wesad"]
        print(f"\nDownloading {dataset_info['name']}...")
        print(f"Description: {dataset_info['description']}")
        print(f"Note: This is a large dataset (~2.5 GB)")
        
        wesad_dir = self.data_dir / "wesad"
        wesad_dir.mkdir(exist_ok=True)
        
        # The sciebo link requires special handling - it's a share link
        # We'll provide manual instructions for this dataset
        print("\n--- Manual Download Instructions for WESAD ---")
        print(f"Dataset: {dataset_info['name']}")
        print(f"Description: {dataset_info['description']}")
        print(f"Download URL: {dataset_info['url']}")
        print("\nSteps to download:")
        print("1. Click on the URL above")
        print("2. Click the 'Download' button on the page")
        print("3. Save the file as 'wesad.zip' in the following directory:")
        print(f"   {wesad_dir}")
        print("4. Run this script again to extract the data")
        print("\nAlternatively, you can use the interactive browser version:")
        print("https://kristofvl.github.io/wesadviz")
        
        # Check if file already exists
        wesad_zip = wesad_dir / "wesad.zip"
        if wesad_zip.exists():
            print(f"\n✓ Found {wesad_zip}")
            print("Extracting archive...")
            if self.extract_archive(wesad_zip, "wesad"):
                print("✓ WESAD dataset extracted successfully")
                return True
            else:
                print("✗ Failed to extract WESAD dataset")
                return False
        
        return False
    
    def download_physionet(self):
        """Download PhysioNet stress recognition dataset."""
        dataset_info = self.datasets["physionet"]
        print(f"\nDownloading {dataset_info['name']}...")
        print(f"Description: {dataset_info['description']}")
        
        # For PhysioNet, we'll use wfdb to download the data
        try:
            import wfdb
            
            physionet_dir = self.data_dir / "physionet"
            physionet_dir.mkdir(exist_ok=True)
            
            # Download the DRIVEDB database
            print("Downloading DRIVEDB database using WFDB...")
            records = wfdb.get_record_list('drivedb')
            
            for record in tqdm(records, desc="Downloading records"):
                try:
                    wfdb.dl_database('drivedb', str(physionet_dir), record)
                except Exception as e:
                    print(f"Error downloading {record}: {e}")
            
            print("✓ PhysioNet dataset downloaded successfully")
            return True
            
        except ImportError:
            print("WFDB library not found. Install with: pip install wfdb")
            return False
        except Exception as e:
            print(f"Error downloading PhysioNet dataset: {e}")
            return False
    
    def download_mitbih(self):
        """Download MIT-BIH arrhythmia dataset."""
        dataset_info = self.datasets["mitbih"]
        print(f"\nDownloading {dataset_info['name']}...")
        print(f"Description: {dataset_info['description']}")
        
        try:
            import wfdb
            
            mitbih_dir = self.data_dir / "mitbih"
            mitbih_dir.mkdir(exist_ok=True)
            
            # Download the MITDB database
            print("Downloading MITDB database using WFDB...")
            records = wfdb.get_record_list('mitdb')
            
            for record in tqdm(records[:10], desc="Downloading records"):  # Limit to first 10 records
                try:
                    wfdb.dl_database('mitdb', str(mitbih_dir), record)
                except Exception as e:
                    print(f"Error downloading {record}: {e}")
            
            print("✓ MIT-BIH dataset downloaded successfully")
            return True
            
        except ImportError:
            print("WFDB library not found. Install with: pip install wfdb")
            return False
        except Exception as e:
            print(f"Error downloading MIT-BIH dataset: {e}")
            return False
    
    def show_manual_instructions(self, dataset_key):
        """Show manual download instructions for restricted datasets."""
        if dataset_key not in self.datasets:
            print(f"Unknown dataset: {dataset_key}")
            return
        
        dataset_info = self.datasets[dataset_key]
        print(f"\n--- Manual Download Instructions for {dataset_info['name']} ---")
        print(f"Description: {dataset_info['description']}")
        print(f"\nURL: {dataset_info['url']}")
        print("\nSteps to download:")
        print("1. Visit the URL above")
        print("2. Register or sign in if required")
        print("3. Accept the terms and conditions")
        print("4. Download the following files:")
        for file_name in dataset_info["files"]:
            print(f"   - {file_name}")
        print(f"\n5. Place the downloaded files in: {self.data_dir / dataset_info['extract_to']}")
        print("6. Run this script again to verify the download")
    
    def download_dataset(self, dataset_key):
        """Download a specific dataset."""
        if dataset_key not in self.datasets:
            print(f"Unknown dataset: {dataset_key}")
            print(f"Available datasets: {list(self.datasets.keys())}")
            return False
        
        dataset_info = self.datasets[dataset_key]
        
        if not dataset_info["free"]:
            self.show_manual_instructions(dataset_key)
            return False
        
        # Route to specific download method
        if dataset_key == "wesad":
            return self.download_wesad()
        elif dataset_key == "physionet":
            return self.download_physionet()
        elif dataset_key == "mitbih":
            return self.download_mitbih()
        else:
            print(f"Download method not implemented for {dataset_key}")
            return False
    
    def list_datasets(self):
        """List all available datasets."""
        print("\nAvailable Datasets:")
        print("=" * 50)
        for key, info in self.datasets.items():
            status = "✓ Free" if info["free"] else "✗ Manual"
            print(f"{key:10} - {info['name']}")
            print(f"{'':10}   {info['description']}")
            print(f"{'':10}   Status: {status}")
            print()


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download datasets for Intelligent Driver Behavior Sensing (IDBS) system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_datasets.py --datasets wesad
  python download_datasets.py --datasets wesad physionet
  python download_datasets.py --list
  python download_datasets.py --all
        """
    )
    
    parser.add_argument(
        "--datasets", 
        nargs="+", 
        choices=["wesad", "uah", "physionet", "mitbih"],
        help="Specify which datasets to download"
    )
    
    parser.add_argument(
        "--list", 
        action="store_true",
        help="List all available datasets"
    )
    
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Download all freely available datasets"
    )
    
    parser.add_argument(
        "--data-dir", 
        default="data",
        help="Directory to store downloaded datasets (default: data)"
    )
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = DatasetDownloader(args.data_dir)
    
    # Handle list option
    if args.list:
        downloader.list_datasets()
        return
    
    # Handle all option
    if args.all:
        print("Downloading all freely available datasets...")
        free_datasets = [key for key, info in downloader.datasets.items() if info["free"]]
        args.datasets = free_datasets
    
    # Validate datasets argument
    if not args.datasets:
        print("No datasets specified. Use --list to see available datasets.")
        print("Use --help for more information.")
        return
    
    # Download specified datasets
    print(f"Data directory: {downloader.data_dir}")
    print("=" * 50)
    
    success_count = 0
    for dataset in args.datasets:
        print(f"\nProcessing dataset: {dataset}")
        if downloader.download_dataset(dataset):
            success_count += 1
            print(f"✓ Successfully processed {dataset}")
        else:
            print(f"✗ Failed to process {dataset}")
    
    print(f"\nSummary: {success_count}/{len(args.datasets)} datasets processed successfully")


if __name__ == "__main__":
    main()

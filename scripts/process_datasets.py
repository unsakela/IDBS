"""
Data Processing Script for IDBS System

This script processes and prepares datasets for the Intelligent Driver Behavior
Sensing system, including WESAD and State Farm Distracted Driver datasets.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
import pickle
import warnings
warnings.filterwarnings('ignore')


class DatasetProcessor:
    """Main class for processing IDBS datasets."""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(exist_ok=True)
        
    def process_wesad(self, extract_to="data/wesad"):
        """Process WESAD dataset."""
        print("Processing WESAD dataset...")
        
        wesad_dir = Path(extract_to)
        if not wesad_dir.exists():
            print(f"WESAD directory not found: {wesad_dir}")
            return False
        
        # Look for subject directories
        subject_dirs = [d for d in wesad_dir.iterdir() if d.is_dir() and d.name.startswith('S')]
        if not subject_dirs:
            print("No subject directories found in WESAD folder")
            print("Please extract the WESAD zip file to the data/wesad/ directory")
            return False
        
        print(f"Found {len(subject_dirs)} subject directories")
        
        processed_wesad = self.processed_dir / "wesad"
        processed_wesad.mkdir(exist_ok=True)
        
        all_data = []
        all_labels = []
        subject_info = []
        
        for subject_dir in sorted(subject_dirs):
            subject_id = subject_dir.name
            print(f"Processing {subject_id}...")
            
            # Look for the main data file
            data_file = subject_dir / f"{subject_id}.pkl"
            if not data_file.exists():
                print(f"Data file not found: {data_file}")
                continue
            
            try:
                with open(data_file, 'rb') as f:
                    subject_data = pickle.load(f, encoding='latin1')
                
                # Extract sensor data for different conditions
                conditions = ['baseline', 'stress', 'amusement']
                
                for condition in conditions:
                    if condition in subject_data['signal']:
                        # Extract chest sensor data (ECG, EDA, EMG, etc.)
                        chest_data = subject_data['signal'][condition]['chest']
                        
                        # Extract wrist sensor data
                        wrist_data = subject_data['signal'][condition]['wrist']
                        
                        # Create a unified sample
                        sample = {
                            'subject_id': subject_id,
                            'condition': condition,
                            'chest_ecg': chest_data['ECG'],
                            'chest_eda': chest_data['EDA'],
                            'chest_emg': chest_data['EMG'],
                            'chest_temp': chest_data['Temp'],
                            'chest_acc': chest_data['ACC'],
                            'wrist_bvp': wrist_data['BVP'],
                            'wrist_eda': wrist_data['EDA'],
                            'wrist_temp': wrist_data['TEMP'],
                            'wrist_acc': wrist_data['ACC']
                        }
                        
                        all_data.append(sample)
                        all_labels.append(condition)
                        subject_info.append({
                            'subject_id': subject_id,
                            'condition': condition,
                            'num_samples': len(chest_data['ECG'])
                        })
                
            except Exception as e:
                print(f"Error processing {subject_id}: {e}")
                continue
        
        if all_data:
            # Save processed data
            processed_data = {
                'samples': all_data,
                'labels': all_labels,
                'subject_info': subject_info,
                'num_samples': len(all_data),
                'conditions': list(set(all_labels))
            }
            
            output_file = processed_wesad / "wesad_processed.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(processed_data, f)
            
            # Save summary as CSV
            summary_df = pd.DataFrame(subject_info)
            summary_df.to_csv(processed_wesad / "wesad_summary.csv", index=False)
            
            print(f"✓ WESAD data processed and saved to {output_file}")
            print(f"  Total samples: {len(all_data)}")
            print(f"  Conditions: {processed_data['conditions']}")
            
            return True
        
        return False
    
    def process_state_farm(self, extract_to="data/state_farm"):
        """Process State Farm Distracted Driver dataset."""
        print("Processing State Farm Distracted Driver dataset...")
        
        sf_dir = Path(extract_to)
        if not sf_dir.exists():
            print(f"State Farm directory not found: {sf_dir}")
            print("Please extract the State Farm zip file to the data/state_farm/ directory")
            return False
        
        # Look for image directories
        train_dir = sf_dir / "imgs" / "train"
        test_dir = sf_dir / "imgs" / "test"
        
        if not train_dir.exists():
            print(f"Training directory not found: {train_dir}")
            return False
        
        processed_sf = self.processed_dir / "state_farm"
        processed_sf.mkdir(exist_ok=True)
        
        # Get class information
        class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
        classes = sorted([d.name for d in class_dirs])
        
        print(f"Found {len(classes)} classes: {classes}")
        
        # Process training data
        train_data = []
        for class_name in classes:
            class_dir = train_dir / class_name
            images = list(class_dir.glob("*.jpg"))
            
            print(f"Processing {class_name}: {len(images)} images")
            
            for img_path in images:
                train_data.append({
                    'image_path': str(img_path),
                    'class': class_name,
                    'class_id': classes.index(class_name)
                })
        
        # Process test data if available
        test_data = []
        if test_dir.exists():
            test_images = list(test_dir.glob("*.jpg"))
            print(f"Processing test data: {len(test_images)} images")
            
            for img_path in test_images:
                test_data.append({
                    'image_path': str(img_path),
                    'filename': img_path.name
                })
        
        # Save processed data
        processed_data = {
            'train_data': train_data,
            'test_data': test_data,
            'classes': classes,
            'num_classes': len(classes),
            'num_train_samples': len(train_data),
            'num_test_samples': len(test_data)
        }
        
        output_file = processed_sf / "state_farm_processed.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(processed_data, f)
        
        # Save as CSV for easy inspection
        train_df = pd.DataFrame(train_data)
        train_df.to_csv(processed_sf / "state_farm_train.csv", index=False)
        
        if test_data:
            test_df = pd.DataFrame(test_data)
            test_df.to_csv(processed_sf / "state_farm_test.csv", index=False)
        
        print(f"✓ State Farm data processed and saved to {output_file}")
        print(f"  Training samples: {len(train_data)}")
        print(f"  Test samples: {len(test_data)}")
        print(f"  Classes: {classes}")
        
        return True
    
    def create_data_splits(self, dataset_name, test_size=0.2, val_size=0.1, random_state=42):
        """Create train/val/test splits for processed datasets."""
        print(f"Creating data splits for {dataset_name}...")
        
        processed_file = self.processed_dir / dataset_name / f"{dataset_name}_processed.pkl"
        
        if not processed_file.exists():
            print(f"Processed data file not found: {processed_file}")
            return False
        
        with open(processed_file, 'rb') as f:
            data = pickle.load(f)
        
        from sklearn.model_selection import train_test_split
        
        if dataset_name == "wesad":
            # For WESAD, split by subjects to avoid data leakage
            subjects = list(set([s['subject_id'] for s in data['subject_info']]))
            train_subjects, test_subjects = train_test_split(subjects, test_size=test_size, random_state=random_state)
            train_subjects, val_subjects = train_test_split(train_subjects, test_size=val_size/(1-test_size), random_state=random_state)
            
            # Create splits
            train_indices = [i for i, s in enumerate(data['subject_info']) if s['subject_id'] in train_subjects]
            val_indices = [i for i, s in enumerate(data['subject_info']) if s['subject_id'] in val_subjects]
            test_indices = [i for i, s in enumerate(data['subject_info']) if s['subject_id'] in test_subjects]
            
        elif dataset_name == "state_farm":
            # For State Farm, random split
            train_data, temp_data = train_test_split(data['train_data'], test_size=test_size+val_size, random_state=random_state)
            val_data, test_data = train_test_split(temp_data, test_size=test_size/(test_size+val_size), random_state=random_state)
            
            train_indices = list(range(len(train_data)))
            val_indices = list(range(len(train_data), len(train_data) + len(val_data)))
            test_indices = list(range(len(train_data) + len(val_data), len(data['train_data'])))
        
        # Save splits
        splits = {
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices,
            'dataset_info': {
                'name': dataset_name,
                'total_samples': len(data['samples'] if 'samples' in data else data['train_data']),
                'train_size': len(train_indices),
                'val_size': len(val_indices),
                'test_size': len(test_indices)
            }
        }
        
        splits_file = self.processed_dir / dataset_name / f"{dataset_name}_splits.pkl"
        with open(splits_file, 'wb') as f:
            pickle.dump(splits, f)
        
        print(f"✓ Data splits saved to {splits_file}")
        print(f"  Train: {len(train_indices)} samples")
        print(f"  Val: {len(val_indices)} samples")
        print(f"  Test: {len(test_indices)} samples")
        
        return True
    
    def generate_summary_report(self):
        """Generate a summary report of all processed datasets."""
        print("Generating summary report...")
        
        report = {
            'datasets': {},
            'total_datasets': 0
        }
        
        for dataset_dir in self.processed_dir.iterdir():
            if dataset_dir.is_dir():
                dataset_name = dataset_dir.name
                processed_file = dataset_dir / f"{dataset_name}_processed.pkl"
                
                if processed_file.exists():
                    with open(processed_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    report['datasets'][dataset_name] = {
                        'num_samples': data.get('num_samples', len(data.get('samples', data.get('train_data', [])))),
                        'classes': data.get('conditions', data.get('classes', [])),
                        'num_classes': len(data.get('conditions', data.get('classes', []))),
                        'processed_file': str(processed_file)
                    }
                    
                    report['total_datasets'] += 1
        
        # Save report
        report_file = self.processed_dir / "summary_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Summary report saved to {report_file}")
        print(f"  Total datasets processed: {report['total_datasets']}")
        
        return report


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process datasets for Intelligent Driver Behavior Sensing (IDBS) system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_datasets.py --datasets wesad
  python process_datasets.py --datasets wesad state_farm
  python process_datasets.py --all
  python process_datasets.py --splits wesad
        """
    )
    
    parser.add_argument(
        "--datasets", 
        nargs="+", 
        choices=["wesad", "state_farm"],
        help="Specify which datasets to process"
    )
    
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Process all available datasets"
    )
    
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["wesad", "state_farm"],
        help="Create data splits for specified datasets"
    )
    
    parser.add_argument(
        "--data-dir", 
        default="data",
        help="Directory containing raw datasets (default: data)"
    )
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = DatasetProcessor(args.data_dir)
    
    # Handle all option
    if args.all:
        args.datasets = ["wesad", "state_farm"]
    
    # Process datasets
    if args.datasets:
        print(f"Data directory: {processor.data_dir}")
        print("=" * 50)
        
        success_count = 0
        for dataset in args.datasets:
            print(f"\nProcessing dataset: {dataset}")
            if dataset == "wesad":
                if processor.process_wesad():
                    success_count += 1
                    print(f"✓ Successfully processed {dataset}")
                else:
                    print(f"✗ Failed to process {dataset}")
            elif dataset == "state_farm":
                if processor.process_state_farm():
                    success_count += 1
                    print(f"✓ Successfully processed {dataset}")
                else:
                    print(f"✗ Failed to process {dataset}")
        
        print(f"\nSummary: {success_count}/{len(args.datasets)} datasets processed successfully")
    
    # Create data splits
    if args.splits:
        print(f"\nCreating data splits...")
        for dataset in args.splits:
            if processor.create_data_splits(dataset):
                print(f"✓ Created splits for {dataset}")
            else:
                print(f"✗ Failed to create splits for {dataset}")
    
    # Generate summary report
    if args.datasets or args.splits:
        report = processor.generate_summary_report()
        print(f"\n✓ Processing complete!")
        print(f"Processed datasets: {list(report['datasets'].keys())}")


if __name__ == "__main__":
    main()

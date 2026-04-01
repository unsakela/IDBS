"""
GitHub Driving Data Preprocessing Script - Fixed for (N, 500, 64) output
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
import warnings
import argparse
warnings.filterwarnings('ignore')


class GitHubDrivingPreprocessor:
    """
    Preprocessor for GitHub Driving dataset.
    Processes accelerometer and gyroscope data with sliding window analysis.
    """
    
    def __init__(self, sample_rate: float = 50.0, window_size: int = 500, stride: int = 250):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.stride = stride
        
        # Flexible column mapping
        self.COLUMN_ALIASES = {
            'accel_x': ['AccX', 'acc_x', 'ax', 'Ax', 'accelerometer_x', 'accel_x', 'x', 'aceleracaoLinear'],
            'accel_y': ['AccY', 'acc_y', 'ay', 'Ay', 'accelerometer_y', 'accel_y', 'y'],
            'accel_z': ['AccZ', 'acc_z', 'az', 'Az', 'accelerometer_z', 'accel_z', 'z'],
            'gyro_x': ['GyroX', 'gyr_x', 'gx', 'Gx', 'gyroscope_x', 'gyro_x', 'giroscopio'],
            'gyro_y': ['GyroY', 'gyr_y', 'gy', 'Gy', 'gyroscope_y', 'gyro_y'],
            'gyro_z': ['GyroZ', 'gyr_z', 'gz', 'Gz', 'gyroscope_z', 'gyro_z'],
            'mag_x': ['MagX', 'mag_x', 'mx', 'Mx', 'magnetometer_x', 'campoMagnetico'],
            'mag_y': ['MagY', 'mag_y', 'my', 'My', 'magnetometer_y'],
            'mag_z': ['MagZ', 'mag_z', 'mz', 'Mz', 'magnetometer_z'],
        }
        
        # Label mapping
        self.label_map = {
            'normal': 0,
            'safe': 0,
            'aggressive': 1,
            'risky': 2,
            'dangerous': 2
        }
    
    def find_column(self, df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
        """Find column name from list of aliases."""
        for alias in aliases:
            if alias in df.columns:
                return alias
        return None
    
    def map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map various column naming conventions to standard names."""
        mapped_df = df.copy()
        
        # Use flexible column finding
        accel_x = self.find_column(df, self.COLUMN_ALIASES['accel_x'])
        accel_y = self.find_column(df, self.COLUMN_ALIASES['accel_y'])
        accel_z = self.find_column(df, self.COLUMN_ALIASES['accel_z'])
        gyro_x = self.find_column(df, self.COLUMN_ALIASES['gyro_x'])
        gyro_y = self.find_column(df, self.COLUMN_ALIASES['gyro_y'])
        gyro_z = self.find_column(df, self.COLUMN_ALIASES['gyro_z'])
        
        # Create standardized columns
        if accel_x:
            mapped_df['accel_x'] = df[accel_x]
        if accel_y:
            mapped_df['accel_y'] = df[accel_y]
        if accel_z:
            mapped_df['accel_z'] = df[accel_z]
        if gyro_x:
            mapped_df['gyro_x'] = df[gyro_x]
        if gyro_y:
            mapped_df['gyro_y'] = df[gyro_y]
        if gyro_z:
            mapped_df['gyro_z'] = df[gyro_z]
        
        return mapped_df
    
    def sliding_window(self, data: np.ndarray) -> np.ndarray:
        """Create sliding windows from signal."""
        if len(data) < self.window_size:
            return np.array([data])
        
        windows = []
        for start in range(0, len(data) - self.window_size + 1, self.stride):
            end = start + self.window_size
            if end <= len(data):
                windows.append(data[start:end])
        
        return np.array(windows)
    
    def extract_time_domain_features(self, signal: np.ndarray) -> np.ndarray:
        """Extract time-domain features from signal."""
        features = []
        
        # Basic statistics
        features.append(np.mean(signal))
        features.append(np.std(signal))
        features.append(np.min(signal))
        features.append(np.max(signal))
        features.append(np.max(signal) - np.min(signal))  # Range
        features.append(np.sqrt(np.mean(signal**2)))  # RMS
        features.append(skew(signal))
        features.append(kurtosis(signal))
        
        # Zero crossing rate
        sign_changes = np.diff(np.sign(signal))
        zero_crossings = np.where(sign_changes != 0)[0]
        features.append(len(zero_crossings) / len(signal))
        
        # Peak detection
        try:
            peaks, _ = find_peaks(signal, height=np.std(signal))
        except:
            peaks = np.array([])
        features.append(len(peaks))
        
        # Additional features
        features.append(np.sum(np.abs(signal)))
        features.append(np.sum(np.abs(signal)) / len(signal))
        features.append(np.percentile(signal, 25))
        features.append(np.percentile(signal, 50))
        features.append(np.percentile(signal, 75))
        
        # Fill to 32 features
        while len(features) < 32:
            features.append(0.0)
        
        return np.array(features[:32])
    
    def extract_frequency_domain_features(self, signal: np.ndarray) -> np.ndarray:
        """Extract frequency-domain features from signal."""
        features = []
        
        # FFT
        fft_vals = np.fft.fft(signal)
        fft_magnitude = np.abs(fft_vals)
        
        # Spectral features
        features.append(np.mean(fft_magnitude))
        features.append(np.std(fft_magnitude))
        features.append(np.max(fft_magnitude))
        features.append(np.sum(fft_magnitude**2))
        
        # Spectral entropy
        total_energy = np.sum(fft_magnitude**2)
        if total_energy > 0:
            spectral_psd = fft_magnitude**2 / total_energy
            spectral_entropy = -np.sum(spectral_psd * np.log2(spectral_psd + 1e-10))
            features.append(spectral_entropy)
        else:
            features.append(0)
        
        # Peak frequency
        peak_idx = np.argmax(fft_magnitude)
        fft_freq = np.fft.fftfreq(len(signal), 1/self.sample_rate)
        features.append(fft_freq[peak_idx])
        
        # Fill to 32 features
        while len(features) < 32:
            features.append(0.0)
        
        return np.array(features[:32])
    
    def extract_features(self, signal: np.ndarray) -> np.ndarray:
        """Extract 64 features from signal window."""
        time_features = self.extract_time_domain_features(signal)
        freq_features = self.extract_frequency_domain_features(signal)
        
        combined_features = np.concatenate([time_features, freq_features])
        
        if len(combined_features) < 64:
            padding = np.zeros(64 - len(combined_features))
            combined_features = np.concatenate([combined_features, padding])
        elif len(combined_features) > 64:
            combined_features = combined_features[:64]
        
        return combined_features
    
    def extract_derived_features(self, accel_data: np.ndarray, gyro_data: np.ndarray) -> np.ndarray:
        """Extract derived features from accelerometer and gyroscope data."""
        features = []
        
        # Magnitude features
        accel_magnitude = np.sqrt(np.sum(accel_data**2, axis=1))
        gyro_magnitude = np.sqrt(np.sum(gyro_data**2, axis=1))
        
        features.append(np.mean(accel_magnitude))
        features.append(np.std(accel_magnitude))
        features.append(np.mean(gyro_magnitude))
        features.append(np.std(gyro_magnitude))
        
        return np.array(features)
    
    def process_csv_file(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Process a single CSV file."""
        try:
            # Load CSV
            df = pd.read_csv(csv_path)
            
            # Map column names
            df = self.map_columns(df)
            
            # Check if required columns are present
            required_columns = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                # If missing gyroscope data, try accelerometer-only processing
                if all(col in missing_columns for col in ['gyro_x', 'gyro_y', 'gyro_z']):
                    if 'accel_x' in df.columns and 'accel_y' in df.columns and 'accel_z' in df.columns:
                        print(f"    Processing with accelerometer data only")
                        # Use only accelerometer data
                        accel_data = df[['accel_x', 'accel_y', 'accel_z']].values
                        
                        # Infer label from filename (default to normal=0)
                        filename = Path(csv_path).stem.lower()
                        label = 0  # Default to normal
                        
                        # Try to infer specific labels from filename
                        if 'aggressive' in filename:
                            label = 1
                        elif 'risky' in filename or 'dangerous' in filename:
                            label = 2
                        elif 'normal' in filename or 'safe' in filename:
                            label = 0
                        
                        # Create sliding windows
                        all_features = []
                        all_labels = []
                        
                        for start in range(0, len(df) - self.window_size + 1, self.stride):
                            end = start + self.window_size
                            
                            # Get window data
                            window_accel = accel_data[start:end]
                            
                            # Extract features for each accelerometer channel
                            window_features = []
                            
                            for channel_idx in range(3):
                                channel_features = self.extract_features(window_accel[:, channel_idx])
                                window_features.extend(channel_features)
                            
                            # Add derived features (repeat to fill 64)
                            derived_features = self.extract_derived_features(window_accel, window_accel)  # Use same data for both
                            window_features.extend(derived_features)
                            
                            # Add more features to reach 64
                            while len(window_features) < 64:
                                window_features.extend([0.0])  # Pad with zeros
                            
                            # Ensure exactly 64 features
                            window_features = np.array(window_features)
                            if len(window_features) < 64:
                                padding = np.zeros(64 - len(window_features))
                                window_features = np.concatenate([window_features, padding])
                            elif len(window_features) > 64:
                                window_features = window_features[:64]
                            
                            # Tile features across 500 timesteps: (64,) -> (500, 64)
                            tiled_features = np.tile(window_features, (500, 1))
                            all_features.append(tiled_features)
                            all_labels.append(label)
                        
                        return np.array(all_features), np.array(all_labels)
                
                print(f"    Missing columns in {csv_path}: {missing_columns}")
                return np.array([]), np.array([])
            
            # Extract sensor data (ignore magnetometer if present)
            accel_data = df[['accel_x', 'accel_y', 'accel_z']].values
            gyro_data = df[['gyro_x', 'gyro_y', 'gyro_z']].values
            
            # Infer label from filename (default to normal=0)
            filename = Path(csv_path).stem.lower()
            label = 0  # Default to normal
            
            # Try to infer specific labels from filename
            if 'aggressive' in filename:
                label = 1
            elif 'risky' in filename or 'dangerous' in filename:
                label = 2
            elif 'normal' in filename or 'safe' in filename:
                label = 0
            
            # Create sliding windows
            all_features = []
            all_labels = []
            
            for start in range(0, len(df) - self.window_size + 1, self.stride):
                end = start + self.window_size
                
                # Get window data
                window_accel = accel_data[start:end]
                window_gyro = gyro_data[start:end]
                
                # Extract features for each channel
                window_features = []
                
                # Process each accelerometer channel
                for channel_idx in range(3):
                    channel_features = self.extract_features(window_accel[:, channel_idx])
                    window_features.extend(channel_features)
                
                # Process each gyroscope channel
                for channel_idx in range(3):
                    channel_features = self.extract_features(window_gyro[:, channel_idx])
                    window_features.extend(channel_features)
                
                # Add derived features
                derived_features = self.extract_derived_features(window_accel, window_gyro)
                window_features.extend(derived_features)
                
                # Ensure exactly 64 features
                window_features = np.array(window_features)
                if len(window_features) < 64:
                    padding = np.zeros(64 - len(window_features))
                    window_features = np.concatenate([window_features, padding])
                elif len(window_features) > 64:
                    window_features = window_features[:64]
                
                # Tile features across 500 timesteps: (64,) -> (500, 64)
                tiled_features = np.tile(window_features, (500, 1))
                all_features.append(tiled_features)
                all_labels.append(label)
            
            return np.array(all_features), np.array(all_labels)
            
        except Exception as e:
            print(f"    Error processing {csv_path}: {e}")
            return np.array([]), np.array([])
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features."""
        # Reshape to 2D for StandardScaler: (N*500, 64)
        original_shape = features.shape
        features_2d = features.reshape(-1, features.shape[-1])
        
        scaler = StandardScaler()
        normalized_features_2d = scaler.fit_transform(features_2d)
        
        # Reshape back to original 3D shape
        normalized_features = normalized_features_2d.reshape(original_shape)
        return normalized_features
    
    def process_dataset(self, raw_data_dir: str, output_dir: str) -> None:
        """Process entire GitHub driving dataset."""
        print("Processing GitHub Driving dataset...")
        
        # Find all CSV files recursively
        raw_path = Path(raw_data_dir)
        csv_files = []
        
        # Recursively find all CSV files
        for csv_file in raw_path.rglob("*.csv"):
            if 'groundTruth' not in csv_file.name:  # Skip ground truth files
                csv_files.append(str(csv_file))
        
        print(f"Found {len(csv_files)} CSV files")
        
        if not csv_files:
            print("No CSV files found!")
            return
        
        all_features = []
        all_labels = []
        file_metadata = []
        
        # Process each CSV file
        for csv_file in tqdm(csv_files, desc="Processing files"):
            print(f"Processing {Path(csv_file).name}...")
            
            features, labels = self.process_csv_file(csv_file)
            
            if len(features) > 0:
                normalized_features = self.normalize_features(features)
                all_features.append(normalized_features)
                all_labels.extend(labels)
                
                file_metadata.append({
                    'file': Path(csv_file).name,
                    'samples': len(features),
                    'shape': features.shape
                })
                print(f"    Generated {len(features)} windows")
            else:
                print(f"    No valid data generated")
        
        # Combine all data
        if all_features:
            combined_features = np.vstack(all_features)
            combined_labels = np.array(all_labels)
            
            # Create metadata
            metadata = {
                'total_files': len(csv_files),
                'total_windows': len(combined_features),
                'feature_dimension': 500,
                'num_classes': len(np.unique(combined_labels)),
                'label_distribution': dict(zip(*np.unique(combined_labels, return_counts=True))),
                'window_size': self.window_size,
                'stride': self.stride,
                'sample_rate': self.sample_rate,
                'files_processed': file_metadata
            }
            
            # Save processed data
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save main data
            processed_data = {
                'features': combined_features,
                'labels': combined_labels,
                'num_classes': metadata['num_classes'],
                'metadata': metadata
            }
            
            with open(output_path / "processed.pkl", 'wb') as f:
                pickle.dump(processed_data, f)
            
            # Save metadata as CSV
            metadata_df = pd.DataFrame([metadata])
            metadata_df.to_csv(output_path / "metadata.csv", index=False)
            
            print(f"\nDataset Summary:")
            print(f"  Total files: {metadata['total_files']}")
            print(f"  Total windows: {metadata['total_windows']}")
            print(f"  Feature dimension: {metadata['feature_dimension']}")
            print(f"  Number of classes: {metadata['num_classes']}")
            print(f"  Label distribution: {metadata['label_distribution']}")
            
            print(f"✓ GitHub Driving dataset processed successfully!")
            print(f"  Total samples: {len(combined_features)}")
            print(f"  Feature dimension: {metadata['feature_dimension']}")
            print(f"  Number of classes: {metadata['num_classes']}")
            print(f"  Saved to: {output_path}")
        else:
            print("❌ No valid data processed!")


def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description="Preprocess GitHub Driving dataset")
    parser.add_argument("--raw_data_dir", type=str, 
                       default="data/raw/git_driver",
                       help="Directory containing raw GitHub driving data")
    parser.add_argument("--output_dir", type=str, 
                       default="data/processed/github_driving",
                       help="Directory to save processed data")
    parser.add_argument("--sample_rate", type=float, default=50.0,
                       help="Sample rate in Hz")
    parser.add_argument("--window_size", type=int, default=500,
                       help="Window size in samples")
    parser.add_argument("--stride", type=int, default=250,
                       help="Stride between windows")
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = GitHubDrivingPreprocessor(
        sample_rate=args.sample_rate,
        window_size=args.window_size,
        stride=args.stride
    )
    
    print(f"Initialized GitHub Driving Preprocessor:")
    print(f"  Sample rate: {preprocessor.sample_rate} Hz")
    print(f"  Window size: {preprocessor.window_size} samples")
    print(f"  Stride: {preprocessor.stride} samples")
    
    # Process dataset
    preprocessor.process_dataset(args.raw_data_dir, args.output_dir)


if __name__ == "__main__":
    main()
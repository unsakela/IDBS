"""
Mendeley Driving Dataset Preprocessing Script

This script processes Mendeley driving dataset with comprehensive feature extraction,
sliding window analysis, and proper normalization for IDBS system.
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import glob
from scipy import signal
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')


class MendeleyDrivingPreprocessor:
    """
    Preprocessor for Mendeley driving dataset.
    
    Processes accelerometer and gyroscope data with sliding window analysis.
    """
    
    def __init__(self, sample_rate: float = 50.0, window_size: int = 500, stride: int = 250):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.stride = stride
        self.label_map = {'normal': 0, 'aggressive': 1, 'risky': 2}
        
        # Flexible column mapping
        self.COLUMN_ALIASES = {
            'accel_x': ['AccX', 'acc_x', 'ax', 'Ax', 'accelerometer_x', 'accel_x', 'x'],
            'accel_y': ['AccY', 'acc_y', 'ay', 'Ay', 'accelerometer_y', 'accel_y', 'y'],
            'accel_z': ['AccZ', 'acc_z', 'az', 'Az', 'accelerometer_z', 'accel_z', 'z'],
            'gyro_x':  ['GyroX', 'gyr_x', 'gx', 'Gx', 'gyroscope_x', 'gyro_x'],
            'gyro_y':  ['GyroY', 'gyr_y', 'gy', 'Gy', 'gyroscope_y', 'gyro_y'],
            'gyro_z':  ['GyroZ', 'gyr_z', 'gz', 'Gz', 'gyroscope_z', 'gyro_z'],
            'gyro_y': ['gyro_y', 'Gy', 'gy', 'gyroscope_y', 'GyroY'],
            'gyro_z': ['gyro_z', 'Gz', 'gz', 'gyroscope_z', 'GyroZ']
        }
        
        # Label mapping
        self.label_map = {
            'Normal': 0,
            'Aggressive': 1,
            'Risky': 2
        }
        
        print(f"Initialized Mendeley Driving Preprocessor:")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Window size: {window_size} samples")
        print(f"  Stride: {stride} samples")
    
    def find_column(self, df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
        """Find column name from list of aliases."""
        for alias in aliases:
            if alias in df.columns:
                return alias
        return None
    
    def map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map various column naming conventions to standard names.
        """
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
        """
        Create sliding windows from signal.
        
        Args:
            data: Input signal of shape (T,) or (T, C)
            
        Returns:
            Windows of shape (N, window_size) or (N, window_size, C)
        """
        if len(data) < self.window_size:
            return np.array([data])
        
        windows = []
        for start in range(0, len(data) - self.window_size + 1, self.stride):
            end = start + self.window_size
            if end <= len(data):
                windows.append(data[start:end])
        
        return np.array(windows)
    
    def extract_time_domain_features(self, signal: np.ndarray) -> np.ndarray:
        """
        Extract time-domain features from signal.
        
        Args:
            signal: Input signal window
            
        Returns:
            Time-domain features (10 features)
        """
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
        features.append(np.sum(np.abs(np.diff(signal))) / (len(signal) - 1))  # Mean absolute difference
        
        # Zero crossing rate
        sign_changes = np.diff(np.sign(signal))
        zero_crossings = np.where(sign_changes != 0)[0]
        features.append(len(zero_crossings) / len(signal))
        
        return np.array(features)
    
    def extract_frequency_domain_features(self, signal: np.ndarray) -> np.ndarray:
        """
        Extract frequency-domain features from signal.
        
        Args:
            signal: Input signal window
            
        Returns:
            Frequency-domain features (10 features)
        """
        features = []
        
        # FFT
        fft_vals = fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/self.sample_rate)
        power_spectrum = np.abs(fft_vals)**2
        
        # Spectral power in different bands
        # Low frequency (0.5-5 Hz)
        low_mask = (freqs >= 0.5) & (freqs <= 5)
        features.append(np.sum(power_spectrum[low_mask]))
        
        # Mid frequency (5-15 Hz)
        mid_mask = (freqs >= 5) & (freqs <= 15)
        features.append(np.sum(power_spectrum[mid_mask]))
        
        # High frequency (15-25 Hz)
        high_mask = (freqs >= 15) & (freqs <= 25)
        features.append(np.sum(power_spectrum[high_mask]))
        
        # Spectral statistics
        features.append(np.mean(power_spectrum))
        features.append(np.std(power_spectrum))
        features.append(np.max(power_spectrum))
        features.append(np.sum(power_spectrum) / len(power_spectrum))  # Total power
        
        # Spectral entropy
        power_norm = power_spectrum / (np.sum(power_spectrum) + 1e-10)
        features.append(-np.sum(power_norm * np.log2(power_norm + 1e-10)))
        
        return np.array(features)
    
    def extract_derived_features(self, accel_data: np.ndarray, gyro_data: np.ndarray) -> np.ndarray:
        """
        Extract derived features from accelerometer and gyroscope data.
        
        Args:
            accel_data: Accelerometer data of shape (T, 3)
            gyro_data: Gyroscope data of shape (T, 3)
            
        Returns:
            Derived features (4 features)
        """
        features = []
        
        # Magnitude features
        accel_magnitude = np.sqrt(np.sum(accel_data**2, axis=1))
        gyro_magnitude = np.sqrt(np.sum(gyro_data**2, axis=1))
        
        features.append(np.mean(accel_magnitude))
        features.append(np.std(accel_magnitude))
        
        # Jerk (derivative of acceleration)
        jerk = np.diff(accel_magnitude)
        features.append(np.mean(np.abs(jerk)))
        
        # Correlation between accel_x and accel_y
        if accel_data.shape[1] >= 2:
            correlation = np.corrcoef(accel_data[:, 0], accel_data[:, 1])[0, 1]
            features.append(np.nan_to_num(correlation))
        else:
            features.append(0.0)
        
        return np.array(features)
    
    def extract_features(self, signal: np.ndarray) -> np.ndarray:
        """
        Extract comprehensive features from signal window.
        
        Args:
            signal: Input signal window
            
        Returns:
            Feature vector
        """
        # Extract time and frequency domain features
        time_features = self.extract_time_domain_features(signal)
        freq_features = self.extract_frequency_domain_features(signal)
        
        # Combine features
        combined_features = np.concatenate([time_features, freq_features])
        return combined_features
    
    def process_csv_file(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single CSV file.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            Tuple of (features, labels)
        """
        try:
            # Load CSV
            df = pd.read_csv(csv_path)
            
            # Map column names
            df = self.map_columns(df)
            
            # Check if required columns are present
            required_columns = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"    Missing columns in {csv_path}: {missing_columns}")
                return np.array([]), np.array([])
            
            # Extract sensor data
            accel_data = df[['accel_x', 'accel_y', 'accel_z']].values
            gyro_data = df[['gyro_x', 'gyro_y', 'gyro_z']].values
            
            # Infer label from Target column
            if 'Target' in df.columns:
                target_value = df['Target'].iloc[0]
                if target_value == 1:
                    label = 1  # Aggressive
                elif target_value == 2:
                    label = 2  # Risky
                else:
                    label = 0  # Normal
            else:
                # Fallback to filename inference
                filename = Path(csv_path).stem.lower()
                label = 0  # Default to Normal
                
                for label_name, label_value in self.label_map.items():
                    if label_name.lower() in filename:
                        label = label_value
                        break
            
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
        """
        Normalize features.
        
        Args:
            features: Features of shape (N, 500, 64)
            
        Returns:
            Normalized features
        """
        # Reshape to 2D for StandardScaler: (N*500, 64)
        original_shape = features.shape
        features_2d = features.reshape(-1, features.shape[-1])
        
        scaler = StandardScaler()
        normalized_features_2d = scaler.fit_transform(features_2d)
        
        # Reshape back to original 3D shape
        normalized_features = normalized_features_2d.reshape(original_shape)
        return normalized_features
    
    def process_dataset(self, raw_data_dir: str, output_dir: str) -> None:
        """
        Process entire Mendeley driving dataset.
        
        Args:
            raw_data_dir: Directory containing raw Mendeley driving data
            output_dir: Directory to save processed data
        """
        print("Processing Mendeley Driving dataset...")
        
        # Find all CSV files
        raw_path = Path(raw_data_dir)
        csv_files = glob.glob(str(raw_path / "*.csv"))
        
        print(f"Found {len(csv_files)} CSV files")
        
        # DEBUG: Print raw data structure
        print("\n=== RAW DATA INSPECTION ===")
        for f in csv_files:
            try:
                df = pd.read_csv(f)
                print(f"File: {f}")
                print(f"  Shape: {df.shape}")
                print(f"  Columns: {df.columns.tolist()}")
                print(f"  First row: {df.iloc[0].tolist()}")
                print()
            except Exception as e:
                print(f"Error reading {f}: {e}")
        print("=== END INSPECTION ===\n")
        
        if not csv_files:
            print("No CSV files found!")
            return
        
        all_features = []
        all_labels = []
        file_metadata = []
        
        # Process each CSV file
        for csv_file in tqdm(csv_files, desc="Processing files"):
            filename = Path(csv_file).name
            print(f"\nProcessing {filename}...")
            
            features, labels = self.process_csv_file(csv_file)
            
            if len(features) > 0:
                all_features.append(features)
                all_labels.append(labels)
                
                # Store metadata
                file_metadata.append({
                    'filename': filename,
                    'num_windows': len(features),
                    'label_distribution': dict(zip(*np.unique(labels, return_counts=True)))
                })
                
                print(f"    Generated {len(features)} windows")
            else:
                print(f"    No valid windows generated for {filename}")
        
        # Combine all data
        if all_features:
            combined_features = np.vstack(all_features)
            combined_labels = np.hstack(all_labels)
            
            # Normalize features
            normalized_features = self.normalize_features(combined_features)
            
            print(f"\nDataset Summary:")
            print(f"  Total files: {len(file_metadata)}")
            print(f"  Total windows: {len(normalized_features)}")
            print(f"  Feature dimension: {normalized_features.shape[1]}")
            print(f"  Number of classes: {len(np.unique(combined_labels))}")
            print(f"  Label distribution: {dict(zip(*np.unique(combined_labels, return_counts=True)))}")
            
            # Save processed data
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save main data
            processed_data = {
                'features': normalized_features,
                'labels': combined_labels,
                'files': file_metadata,
                'num_samples': len(normalized_features),
                'num_features': normalized_features.shape[1],
                'num_classes': len(np.unique(combined_labels)),
                'label_map': self.label_map,
                'window_size': self.window_size,
                'stride': self.stride,
                'sample_rate': self.sample_rate
            }
            
            with open(output_path / "processed.pkl", 'wb') as f:
                pickle.dump(processed_data, f)
            
            # Save metadata as CSV
            metadata_df = pd.DataFrame(file_metadata)
            metadata_df.to_csv(output_path / "metadata.csv", index=False)
            
            print(f"\n✓ Mendeley Driving dataset processed successfully!")
            print(f"  Total samples: {len(normalized_features)}")
            print(f"  Feature dimension: {normalized_features.shape[1]}")
            print(f"  Number of classes: {len(np.unique(combined_labels))}")
            print(f"  Saved to: {output_path}")
        else:
            print("✗ No data processed successfully!")


def main():
    """Main function to run Mendeley driving preprocessing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process Mendeley driving dataset")
    parser.add_argument("--raw_data_dir", type=str, default="data/raw/mendeley_driving",
                       help="Directory containing raw Mendeley driving data")
    parser.add_argument("--output_dir", type=str, default="data/processed/mendeley_driving",
                       help="Directory to save processed data")
    parser.add_argument("--sample_rate", type=float, default=50.0,
                       help="Sampling rate in Hz")
    parser.add_argument("--window_size", type=int, default=500,
                       help="Window size in samples")
    parser.add_argument("--stride", type=int, default=250,
                       help="Stride between windows")
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = MendeleyDrivingPreprocessor(
        sample_rate=args.sample_rate,
        window_size=args.window_size,
        stride=args.stride
    )
    
    # Process dataset
    preprocessor.process_dataset(args.raw_data_dir, args.output_dir)


if __name__ == "__main__":
    main()

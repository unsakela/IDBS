"""
Evaluation Script for IDBS System

This script evaluates trained models on test sets with comprehensive metrics,
confusion matrices, ROC curves, and detailed performance analysis.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import timm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.sensor.tcn import TCN
from models.fusion.cross_attention import CrossModalAttention
from training.train_fusion import CrossModalAttentionFusion
from data.loaders.wesad_loader import create_wesad_dataloaders
from data.loaders.statefarm_loader import create_statefarm_dataloaders


class ModelEvaluator:
    """
    Comprehensive evaluator for IDBS models.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize Model Evaluator.
        
        Args:
            device: Device to use for evaluation (cuda/cpu)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Results storage
        self.results = {}
        
    def load_tcn_model(self, checkpoint_path: str, input_size: int = 64, num_classes: int = 4) -> TCN:
        """
        Load trained TCN model.
        
        Args:
            checkpoint_path: Path to TCN checkpoint
            input_size: Input feature dimension
            num_classes: Number of output classes
            
        Returns:
            Loaded TCN model
        """
        model = TCN(
            input_size=input_size,
            num_channels=[64, 128, 256],
            kernel_size=3,
            dropout=0.2,
            output_size=num_classes,
            return_sequences=False
        ).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✓ Loaded TCN model from {checkpoint_path}")
        return model
    
    def load_vision_model(self, checkpoint_path: str, num_classes: int = 10) -> nn.Module:
        """
        Load trained EfficientNetV2-S model.
        
        Args:
            checkpoint_path: Path to vision checkpoint
            num_classes: Number of output classes
            
        Returns:
            Loaded vision model
        """
        model = timm.create_model(
            'efficientnetv2_s',
            pretrained=False,
            num_classes=num_classes,
            in_chans=3
        ).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✓ Loaded Vision model from {checkpoint_path}")
        return model
    
    def load_fusion_model(self, checkpoint_path: str, 
                        num_behavior_classes: int = 10, 
                        num_risk_classes: int = 2) -> CrossModalAttentionFusion:
        """
        Load trained fusion model.
        
        Args:
            checkpoint_path: Path to fusion checkpoint
            num_behavior_classes: Number of behavior classes
            num_risk_classes: Number of risk classes
            
        Returns:
            Loaded fusion model
        """
        model = CrossModalAttentionFusion(
            tcn_input_size=64,
            tcn_channels=[64, 128, 256],
            tcn_kernel_size=3,
            vision_model_name='efficientnetv2_s',
            vision_feature_dim=1280,
            num_behavior_classes=num_behavior_classes,
            num_risk_classes=num_risk_classes,
            fusion_dim=512,
            dropout=0.2
        ).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✓ Loaded Fusion model from {checkpoint_path}")
        return model
    
    def evaluate_tcn(self, model: TCN, test_loader: DataLoader, 
                    class_names: List[str] = None) -> Dict:
        """
        Evaluate TCN model on test set.
        
        Args:
            model: Trained TCN model
            test_loader: Test data loader
            class_names: List of class names
            
        Returns:
            Evaluation results dictionary
        """
        print("Evaluating TCN model...")
        
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in tqdm(test_loader, desc="TCN Evaluation"):
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Reshape for TCN
                features = features.unsqueeze(1)
                
                outputs = model(features)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1_macro = f1_score(all_labels, all_predictions, average='macro')
        f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
        precision_macro = precision_score(all_labels, all_predictions, average='macro')
        recall_macro = recall_score(all_labels, all_predictions, average='macro')
        
        # Classification report
        report = classification_report(all_labels, all_predictions, 
                                   target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        results = {
            'model_type': 'TCN',
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': all_predictions.tolist(),
            'probabilities': all_probabilities.tolist(),
            'labels': all_labels.tolist(),
            'class_names': class_names
        }
        
        print(f"TCN Results - Accuracy: {accuracy:.4f}, F1 (weighted): {f1_weighted:.4f}")
        return results
    
    def evaluate_vision(self, model: nn.Module, test_loader: DataLoader,
                       class_names: List[str] = None) -> Dict:
        """
        Evaluate vision model on test set.
        
        Args:
            model: Trained vision model
            test_loader: Test data loader
            class_names: List of class names
            
        Returns:
            Evaluation results dictionary
        """
        print("Evaluating Vision model...")
        
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Vision Evaluation"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1_macro = f1_score(all_labels, all_predictions, average='macro')
        f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
        precision_macro = precision_score(all_labels, all_predictions, average='macro')
        recall_macro = recall_score(all_labels, all_predictions, average='macro')
        
        # Classification report
        report = classification_report(all_labels, all_predictions,
                                   target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        results = {
            'model_type': 'Vision',
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': all_predictions.tolist(),
            'probabilities': all_probabilities.tolist(),
            'labels': all_labels.tolist(),
            'class_names': class_names
        }
        
        print(f"Vision Results - Accuracy: {accuracy:.4f}, F1 (weighted): {f1_weighted:.4f}")
        return results
    
    def evaluate_fusion(self, model: CrossModalAttentionFusion, test_loader: DataLoader,
                       behavior_class_names: List[str] = None,
                       risk_class_names: List[str] = None) -> Dict:
        """
        Evaluate fusion model on test set.
        
        Args:
            model: Trained fusion model
            test_loader: Test data loader
            behavior_class_names: List of behavior class names
            risk_class_names: List of risk class names
            
        Returns:
            Evaluation results dictionary
        """
        print("Evaluating Fusion model...")
        
        all_behavior_predictions = []
        all_behavior_probabilities = []
        all_risk_predictions = []
        all_risk_probabilities = []
        all_behavior_labels = []
        all_risk_labels = []
        
        with torch.no_grad():
            for sensor_features, images, behavior_labels, risk_labels in tqdm(test_loader, desc="Fusion Evaluation"):
                sensor_features = sensor_features.to(self.device)
                images = images.to(self.device)
                behavior_labels = behavior_labels.to(self.device)
                risk_labels = risk_labels.to(self.device)
                
                behavior_logits, risk_logits = model(sensor_features, images)
                
                behavior_probs = torch.softmax(behavior_logits, dim=1)
                risk_probs = torch.softmax(risk_logits, dim=1)
                
                behavior_preds = torch.argmax(behavior_logits, dim=1)
                risk_preds = torch.argmax(risk_logits, dim=1)
                
                all_behavior_predictions.extend(behavior_preds.cpu().numpy())
                all_behavior_probabilities.extend(behavior_probs.cpu().numpy())
                all_risk_predictions.extend(risk_preds.cpu().numpy())
                all_risk_probabilities.extend(risk_probs.cpu().numpy())
                all_behavior_labels.extend(behavior_labels.cpu().numpy())
                all_risk_labels.extend(risk_labels.cpu().numpy())
        
        # Convert to numpy arrays
        all_behavior_predictions = np.array(all_behavior_predictions)
        all_behavior_probabilities = np.array(all_behavior_probabilities)
        all_risk_predictions = np.array(all_risk_predictions)
        all_risk_probabilities = np.array(all_risk_probabilities)
        all_behavior_labels = np.array(all_behavior_labels)
        all_risk_labels = np.array(all_risk_labels)
        
        # Behavior metrics
        behavior_accuracy = accuracy_score(all_behavior_labels, all_behavior_predictions)
        behavior_f1_macro = f1_score(all_behavior_labels, all_behavior_predictions, average='macro')
        behavior_f1_weighted = f1_score(all_behavior_labels, all_behavior_predictions, average='weighted')
        behavior_report = classification_report(all_behavior_labels, all_behavior_predictions,
                                             target_names=behavior_class_names, output_dict=True)
        behavior_cm = confusion_matrix(all_behavior_labels, all_behavior_predictions)
        
        # Risk metrics
        risk_accuracy = accuracy_score(all_risk_labels, all_risk_predictions)
        risk_f1_macro = f1_score(all_risk_labels, all_risk_predictions, average='macro')
        risk_f1_weighted = f1_score(all_risk_labels, all_risk_predictions, average='weighted')
        risk_report = classification_report(all_risk_labels, all_risk_predictions,
                                         target_names=risk_class_names, output_dict=True)
        risk_cm = confusion_matrix(all_risk_labels, all_risk_predictions)
        
        # Combined metrics
        combined_f1 = (behavior_f1_weighted + risk_f1_weighted) / 2
        
        results = {
            'model_type': 'Fusion',
            'behavior': {
                'accuracy': behavior_accuracy,
                'f1_macro': behavior_f1_macro,
                'f1_weighted': behavior_f1_weighted,
                'classification_report': behavior_report,
                'confusion_matrix': behavior_cm.tolist(),
                'predictions': all_behavior_predictions.tolist(),
                'probabilities': all_behavior_probabilities.tolist(),
                'labels': all_behavior_labels.tolist(),
                'class_names': behavior_class_names
            },
            'risk': {
                'accuracy': risk_accuracy,
                'f1_macro': risk_f1_macro,
                'f1_weighted': risk_f1_weighted,
                'classification_report': risk_report,
                'confusion_matrix': risk_cm.tolist(),
                'predictions': all_risk_predictions.tolist(),
                'probabilities': all_risk_probabilities.tolist(),
                'labels': all_risk_labels.tolist(),
                'class_names': risk_class_names
            },
            'combined_f1': combined_f1
        }
        
        print(f"Fusion Results - Behavior Acc: {behavior_accuracy:.4f}, Risk Acc: {risk_accuracy:.4f}, Combined F1: {combined_f1:.4f}")
        return results
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], 
                            title: str, save_path: str) -> None:
        """
        Plot and save confusion matrix.
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            title: Plot title
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curves(self, probabilities: np.ndarray, labels: np.ndarray,
                       class_names: List[str], title: str, save_path: str) -> None:
        """
        Plot and save ROC curves for multi-class classification.
        
        Args:
            probabilities: Class probabilities
            labels: True labels
            class_names: List of class names
            title: Plot title
            save_path: Path to save plot
        """
        plt.figure(figsize=(12, 8))
        
        # One-vs-Rest ROC curves
        for i, class_name in enumerate(class_names):
            y_true = (labels == i).astype(int)
            y_score = probabilities[:, i]
            
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)
            
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, results: Dict, output_dir: str) -> None:
        """
        Save evaluation results to files.
        
        Args:
            results: Evaluation results
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results as JSON
        results_file = output_path / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Results saved to {results_file}")
        
        # Generate plots for each model
        for model_name, model_results in results.items():
            if model_name == 'model_type':  # Skip metadata
                continue
                
            if isinstance(model_results, dict) and 'confusion_matrix' in model_results:
                # Single model results (TCN, Vision)
                cm = np.array(model_results['confusion_matrix'])
                class_names = model_results['class_names']
                
                # Confusion matrix
                self.plot_confusion_matrix(
                    cm, class_names,
                    f'{model_name} Confusion Matrix',
                    output_path / f"{model_name.lower()}_confusion_matrix.png"
                )
                
                # ROC curves
                probabilities = np.array(model_results['probabilities'])
                labels = np.array(model_results['labels'])
                
                self.plot_roc_curves(
                    probabilities, labels, class_names,
                    f'{model_name} ROC Curves',
                    output_path / f"{model_name.lower()}_roc_curves.png"
                )
                
            elif isinstance(model_results, dict) and 'behavior' in model_results:
                # Fusion model results
                for task_name, task_results in model_results.items():
                    if task_name in ['behavior', 'risk']:
                        cm = np.array(task_results['confusion_matrix'])
                        class_names = task_results['class_names']
                        
                        # Confusion matrix
                        self.plot_confusion_matrix(
                            cm, class_names,
                            f'Fusion {task_name.title()} Confusion Matrix',
                            output_path / f"fusion_{task_name}_confusion_matrix.png"
                        )
                        
                        # ROC curves
                        probabilities = np.array(task_results['probabilities'])
                        labels = np.array(task_results['labels'])
                        
                        self.plot_roc_curves(
                            probabilities, labels, class_names,
                            f'Fusion {task_name.title()} ROC Curves',
                            output_path / f"fusion_{task_name}_roc_curves.png"
                        )
        
        print(f"✓ Plots saved to {output_path}")
    
    def create_summary_report(self, results: Dict, output_dir: str) -> None:
        """
        Create a summary report with key metrics.
        
        Args:
            results: Evaluation results
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        
        # Create summary DataFrame
        summary_data = []
        
        for model_name, model_results in results.items():
            if model_name == 'model_type':
                continue
                
            if isinstance(model_results, dict) and 'accuracy' in model_results:
                # Single model results
                summary_data.append({
                    'Model': model_results['model_type'],
                    'Task': 'Single',
                    'Accuracy': model_results['accuracy'],
                    'F1 (Weighted)': model_results['f1_weighted'],
                    'F1 (Macro)': model_results['f1_macro']
                })
                
            elif isinstance(model_results, dict) and 'behavior' in model_results:
                # Fusion model results
                behavior_results = model_results['behavior']
                risk_results = model_results['risk']
                
                summary_data.append({
                    'Model': 'Fusion',
                    'Task': 'Behavior',
                    'Accuracy': behavior_results['accuracy'],
                    'F1 (Weighted)': behavior_results['f1_weighted'],
                    'F1 (Macro)': behavior_results['f1_macro']
                })
                
                summary_data.append({
                    'Model': 'Fusion',
                    'Task': 'Risk',
                    'Accuracy': risk_results['accuracy'],
                    'F1 (Weighted)': risk_results['f1_weighted'],
                    'F1 (Macro)': risk_results['f1_macro']
                })
        
        # Create DataFrame and save
        summary_df = pd.DataFrame(summary_data)
        summary_file = output_path / "summary_report.csv"
        summary_df.to_csv(summary_file, index=False)
        
        print(f"✓ Summary report saved to {summary_file}")
        print("\nSummary Report:")
        print(summary_df.to_string(index=False))


def main():
    """Main function to run evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate IDBS models")
    parser.add_argument("--tcn_checkpoint", type=str, default="checkpoints/tcn_best.pt", help="TCN checkpoint path")
    parser.add_argument("--vision_checkpoint", type=str, default="checkpoints/vision_best.pt", help="Vision checkpoint path")
    parser.add_argument("--fusion_checkpoint", type=str, default="checkpoints/fusion_best.pt", help="Fusion checkpoint path")
    parser.add_argument("--wesad_path", type=str, default="data/processed/wesad/wesad_processed.pkl", help="WESAD data path")
    parser.add_argument("--statefarm_dir", type=str, default="data/processed/statefarm", help="State Farm data directory")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(device=args.device)
    
    # Create test data loaders
    print("Creating test data loaders...")
    wesad_train_loader, wesad_val_loader, wesad_test_loader = create_wesad_dataloaders(
        args.wesad_path, batch_size=64, num_workers=4
    )
    
    statefarm_train_loader, statefarm_val_loader, statefarm_test_loader = create_statefarm_dataloaders(
        args.statefarm_dir, batch_size=32, num_workers=4
    )
    
    # Get dataset info
    wesad_info = wesad_train_loader.dataset.get_dataset_info()
    statefarm_info = statefarm_train_loader.dataset.get_dataset_info()
    
    wesad_class_names = list(wesad_info['label_map'].keys())
    statefarm_class_names = statefarm_info['class_names']
    
    # Evaluate models
    results = {}
    
    # TCN Evaluation
    if os.path.exists(args.tcn_checkpoint):
        tcn_model = evaluator.load_tcn_model(args.tcn_checkpoint, num_classes=len(wesad_class_names))
        results['TCN'] = evaluator.evaluate_tcn(tcn_model, wesad_test_loader, wesad_class_names)
    
    # Vision Evaluation
    if os.path.exists(args.vision_checkpoint):
        vision_model = evaluator.load_vision_model(args.vision_checkpoint, num_classes=len(statefarm_class_names))
        results['Vision'] = evaluator.evaluate_vision(vision_model, statefarm_test_loader, statefarm_class_names)
    
    # Fusion Evaluation
    if os.path.exists(args.fusion_checkpoint):
        # Create multimodal test loader (simplified)
        from training.train_fusion import create_multimodal_dataloader
        fusion_test_loader = create_multimodal_dataloader(wesad_test_loader, statefarm_test_loader, batch_size=16)
        
        fusion_model = evaluator.load_fusion_model(
            args.fusion_checkpoint,
            num_behavior_classes=len(statefarm_class_names),
            num_risk_classes=2
        )
        
        risk_class_names = ['Low Risk', 'High Risk']
        results['Fusion'] = evaluator.evaluate_fusion(
            fusion_model, fusion_test_loader, 
            statefarm_class_names, risk_class_names
        )
    
    # Save results
    if results:
        evaluator.save_results(results, args.output_dir)
        evaluator.create_summary_report(results, args.output_dir)
        print(f"\n✓ Evaluation completed! Results saved to {args.output_dir}")
    else:
        print("✗ No models found for evaluation!")


if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import Dict, List, Optional
from sklearn.metrics import confusion_matrix

def plot_training_results(history: Dict, fold_idx: int = 0, save_dir: str = 'results/plots'):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Dictionary containing 'train_loss', 'val_loss', 'train_acc', 'val_acc' lists
        fold_idx: Current fold index (for title/filename)
        save_dir: Directory to save plots
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    if 'val_loss' in history and history['val_loss']:
        plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title(f'Fold {fold_idx+1}: Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Acc')
    if 'val_acc' in history and history['val_acc']:
        plt.plot(epochs, history['val_acc'], 'r-', label='Validation Acc')
    plt.title(f'Fold {fold_idx+1}: Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'fold_{fold_idx+1}_training_curves.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved training curves to {save_path}")


def plot_confusion_matrix_heatmap(y_true, y_pred, classes: List[str], fold_idx: int = 0, save_dir: str = 'results/plots'):
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names
        fold_idx: Current fold index
        save_dir: Directory to save plots
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Fold {fold_idx+1}: Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'fold_{fold_idx+1}_confusion_matrix.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")

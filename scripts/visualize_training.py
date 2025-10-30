#!/usr/bin/env python3
"""
Visualize training history and model performance
"""

import os
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import json
import glob
from datetime import datetime

def find_latest_model():
    """Find the most recent model file"""
    # Try multiple possible model patterns
    model_patterns = [
        '../models/trashformer_finetuned_*.keras',
        '../models/trashformer_best_*.keras', 
        '../models/trashformer_final*.keras',
        '../models/*.keras'
    ]
    
    for pattern in model_patterns:
        model_files = glob.glob(pattern)
        if model_files:
            return max(model_files, key=os.path.getctime)
    
    return None

def plot_training_history(history_dict, save_path='training_history.png'):
    """
    Plot training and validation metrics
    
    Args:
        history_dict: Dictionary with 'loss', 'accuracy', 'val_loss', 'val_accuracy'
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history_dict['loss']) + 1)
    
    # Plot loss
    ax1.plot(epochs, history_dict['loss'], 'b-o', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history_dict['val_loss'], 'r-s', label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, history_dict['accuracy'], 'b-o', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history_dict['val_accuracy'], 'r-s', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add best values
    best_val_acc = max(history_dict['val_accuracy'])
    best_val_epoch = history_dict['val_accuracy'].index(best_val_acc) + 1
    ax2.axhline(y=best_val_acc, color='g', linestyle='--', alpha=0.5)
    ax2.text(0.02, 0.98, f'Best Val Acc: {best_val_acc:.4f} (Epoch {best_val_epoch})',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Training history plot saved to: {save_path}")
    plt.close()

def print_training_summary(history_dict):
    """Print a summary of the training results"""
    print("\n" + "="*70)
    print("📊 TRAINING SUMMARY")
    print("="*70)
    
    epochs_completed = len(history_dict['loss'])
    
    # Final metrics
    final_train_loss = history_dict['loss'][-1]
    final_train_acc = history_dict['accuracy'][-1]
    final_val_loss = history_dict['val_loss'][-1]
    final_val_acc = history_dict['val_accuracy'][-1]
    
    # Best metrics
    best_val_acc = max(history_dict['val_accuracy'])
    best_val_epoch = history_dict['val_accuracy'].index(best_val_acc) + 1
    best_val_loss = min(history_dict['val_loss'])
    
    print(f"\n📈 EPOCHS COMPLETED: {epochs_completed}")
    print(f"\n🎯 FINAL METRICS (Epoch {epochs_completed}):")
    print(f"   Training   → Loss: {final_train_loss:.4f} | Accuracy: {final_train_acc:.4f}")
    print(f"   Validation → Loss: {final_val_loss:.4f} | Accuracy: {final_val_acc:.4f}")
    
    print(f"\n⭐ BEST VALIDATION METRICS:")
    print(f"   Best Accuracy: {best_val_acc:.4f} (Epoch {best_val_epoch})")
    print(f"   Best Loss: {best_val_loss:.4f}")
    
    # Overfitting check
    train_val_diff = final_train_acc - final_val_acc
    if train_val_diff > 0.1:
        print(f"\n⚠️  OVERFITTING DETECTED:")
        print(f"   Training-Validation gap: {train_val_diff:.4f}")
        print(f"   Consider: increasing dropout, adding regularization, or more data augmentation")
    else:
        print(f"\n✅ GOOD MODEL GENERALIZATION:")
        print(f"   Training-Validation gap: {train_val_diff:.4f}")
    
    print("="*70 + "\n")

def load_model_info():
    """Load information about saved models"""
    print("\n" + "="*70)
    print("💾 SAVED MODELS")
    print("="*70 + "\n")
    
    # Look for model files in multiple locations
    model_patterns = [
        '../models/*.keras',
        '../models/*.h5',
        'models/*.keras',
        'models/*.h5'
    ]
    
    model_files = []
    for pattern in model_patterns:
        model_files.extend(glob.glob(pattern))
    
    if not model_files:
        print("❌ No models found in models/ directory")
        print("📁 Checked paths:")
        for pattern in model_patterns:
            print(f"   - {pattern}")
        return
    
    model_files.sort(key=os.path.getctime, reverse=True)
    
    for idx, model_path in enumerate(model_files, 1):
        size_mb = os.path.getsize(model_path) / (1024*1024)
        mod_time = datetime.fromtimestamp(os.path.getctime(model_path))
        print(f"{idx}. {os.path.basename(model_path)}")
        print(f"   Size: {size_mb:.1f} MB")
        print(f"   Created: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

def main():
    """Main function to visualize training results"""
    
    print("\n" + "="*70)
    print("📈 TRASHFORMER - Training Visualization")
    print("="*70)
    
    # Look for history file - try multiple locations
    history_paths = ['../training_history.json', 'training_history.json', './training_history.json']
    history_file = None
    
    for path in history_paths:
        if os.path.exists(path):
            history_file = path
            break
    
    if os.path.exists(history_file):
        print(f"\n✅ Found training history: {history_file}")
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        print_training_summary(history)
        plot_training_history(history)
    else:
        print(f"\n⚠️  No training history file found: {history_file}")
        print("\n💡 To save training history, add this to your training script:")
        print("""
    import json
    
    # After training
    with open('training_history.json', 'w') as f:
        json.dump(history.history, f, indent=2)
        """)
    
    # Show model information
    load_model_info()

if __name__ == "__main__":
    main()


from typing import Dict, Optional, List

def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> None:
    """
    Plot training history
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'val_metric'
        save_path: Path to save plot
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Metric plot
    axes[1].plot(history['val_metric'], label='Validation Metric', 
                linewidth=2, color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Metric')
    axes[1].set_title('Validation Performance')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
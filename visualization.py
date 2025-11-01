"""
Visualization utilities cho Sign2VN
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
import cv2


def visualize_attention(
    attention_weights: List[np.ndarray],
    source_frames: np.ndarray,
    target_words: List[str],
    save_path: Optional[str] = None,
    figsize: tuple = (15, 8)
):
    """
    Visualize attention weights
    
    Args:
        attention_weights: List of attention weights (target_len, source_len)
        source_frames: Number of source frames
        target_words: List of target words
        save_path: Path to save figure
        figsize: Figure size
    """
    # Stack attention weights
    attention_matrix = np.array([att.squeeze() for att in attention_weights])
    
    # Create heatmap
    plt.figure(figsize=figsize)
    
    sns.heatmap(
        attention_matrix,
        xticklabels=[f'Frame {i}' for i in range(attention_matrix.shape[1])],
        yticklabels=target_words,
        cmap='Blues',
        cbar=True,
        square=False
    )
    
    plt.xlabel('Source Frames', fontsize=12)
    plt.ylabel('Target Words', fontsize=12)
    plt.title('Attention Weights Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention visualization saved to {save_path}")
    
    plt.show()


def visualize_landmarks_sequence(
    landmarks: np.ndarray,
    frame_indices: List[int] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (20, 4)
):
    """
    Visualize landmarks tại các frame khác nhau
    
    Args:
        landmarks: (num_frames, num_landmarks * 3)
        frame_indices: List of frame indices to visualize
        save_path: Path to save figure
        figsize: Figure size
    """
    if frame_indices is None:
        # Select evenly spaced frames
        num_frames = min(8, len(landmarks))
        frame_indices = np.linspace(0, len(landmarks)-1, num_frames, dtype=int)
    
    num_plots = len(frame_indices)
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)
    
    if num_plots == 1:
        axes = [axes]
    
    for idx, frame_idx in enumerate(frame_indices):
        ax = axes[idx]
        
        # Extract x, y coordinates
        frame_landmarks = landmarks[frame_idx]
        x_coords = frame_landmarks[0::3]
        y_coords = frame_landmarks[1::3]
        
        # Plot landmarks
        ax.scatter(x_coords, y_coords, s=5, alpha=0.6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_title(f'Frame {frame_idx}')
        ax.axis('off')
    
    plt.suptitle('Landmarks Sequence Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Landmarks visualization saved to {save_path}")
    
    plt.show()


def plot_training_curves(
    history: dict,
    save_path: Optional[str] = None,
    figsize: tuple = (15, 10)
):
    """
    Plot training curves từ history
    
    Args:
        history: Dictionary chứa training history
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train', linewidth=2)
    axes[0, 1].plot(history['val_acc'], label='Validation', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Accuracy', fontsize=11)
    axes[0, 1].set_title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # BLEU Score
    axes[1, 0].plot(history['val_bleu'], linewidth=2, color='green')
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('BLEU Score', fontsize=11)
    axes[1, 0].set_title('Validation BLEU Score', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[1, 1].plot(history['lr'], linewidth=2, color='red')
    axes[1, 1].set_xlabel('Epoch', fontsize=11)
    axes[1, 1].set_ylabel('Learning Rate', fontsize=11)
    axes[1, 1].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()
    
    # Print summary
    best_epoch = np.argmin(history['val_loss'])
    print(f"\nBest model at epoch {best_epoch}:")
    print(f"  Train Loss: {history['train_loss'][best_epoch]:.4f}")
    print(f"  Val Loss: {history['val_loss'][best_epoch]:.4f}")
    print(f"  Val Accuracy: {history['val_acc'][best_epoch]:.4f}")
    print(f"  Val BLEU: {history['val_bleu'][best_epoch]:.4f}")


def visualize_prediction_comparison(
    predictions: List[dict],
    num_samples: int = 10,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 8)
):
    """
    Visualize comparison giữa target và prediction
    
    Args:
        predictions: List of prediction dictionaries
        num_samples: Number of samples to show
        save_path: Path to save figure
        figsize: Figure size
    """
    samples = predictions[:num_samples]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(samples))
    
    # Plot target vs prediction
    for i, sample in enumerate(samples):
        target = sample['target']
        pred = sample['prediction']
        
        # Check if prediction is correct
        is_correct = target == pred
        color = 'green' if is_correct else 'red'
        
        ax.text(0.02, y_pos[i] + 0.2, f"Target:", fontsize=10, fontweight='bold')
        ax.text(0.15, y_pos[i] + 0.2, target, fontsize=10)
        
        ax.text(0.02, y_pos[i] - 0.2, f"Pred:", fontsize=10, fontweight='bold')
        ax.text(0.15, y_pos[i] - 0.2, pred, fontsize=10, color=color)
        
        ax.axhline(y=y_pos[i] + 0.5, color='gray', linestyle='--', alpha=0.3)
    
    ax.set_ylim(-0.5, len(samples))
    ax.set_xlim(0, 1)
    ax.axis('off')
    ax.set_title('Prediction Comparison (Green=Correct, Red=Wrong)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction comparison saved to {save_path}")
    
    plt.show()


def create_video_with_prediction(
    video_path: str,
    prediction_text: str,
    output_path: str,
    fps: Optional[int] = None
):
    """
    Tạo video mới với prediction text overlay
    
    Args:
        video_path: Path to input video
        prediction_text: Predicted text to overlay
        output_path: Path to save output video
        fps: Output FPS (None to use input FPS)
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps is None:
        fps = input_fps
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Creating video with prediction overlay...")
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add text overlay
        # Background rectangle
        cv2.rectangle(frame, (10, height - 60), (width - 10, height - 10), 
                     (0, 0, 0), -1)
        
        # Prediction text
        cv2.putText(frame, f"Prediction: {prediction_text}", 
                   (20, height - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    
    print(f"Video saved to {output_path}")
    print(f"Total frames: {frame_count}")


def analyze_dataset_distribution(
    labels: List[str],
    save_path: Optional[str] = None,
    figsize: tuple = (15, 8)
):
    """
    Phân tích phân bố dataset
    
    Args:
        labels: List of label texts
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Label length distribution
    label_lengths = [len(label.split()) for label in labels]
    axes[0, 0].hist(label_lengths, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Number of Words', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Label Length Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Top words
    all_words = ' '.join(labels).split()
    from collections import Counter
    word_freq = Counter(all_words)
    top_20_words = word_freq.most_common(20)
    
    words, freqs = zip(*top_20_words)
    axes[0, 1].barh(range(len(words)), freqs)
    axes[0, 1].set_yticks(range(len(words)))
    axes[0, 1].set_yticklabels(words)
    axes[0, 1].set_xlabel('Frequency', fontsize=11)
    axes[0, 1].set_title('Top 20 Most Common Words', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    # 3. Unique words count
    unique_words = len(set(all_words))
    total_words = len(all_words)
    
    axes[1, 0].bar(['Unique Words', 'Total Words'], [unique_words, total_words])
    axes[1, 0].set_ylabel('Count', fontsize=11)
    axes[1, 0].set_title('Vocabulary Statistics', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate([unique_words, total_words]):
        axes[1, 0].text(i, v + max([unique_words, total_words])*0.02, 
                       str(v), ha='center', fontweight='bold')
    
    # 4. Character length distribution
    char_lengths = [len(label) for label in labels]
    axes[1, 1].hist(char_lengths, bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 1].set_xlabel('Number of Characters', fontsize=11)
    axes[1, 1].set_ylabel('Frequency', fontsize=11)
    axes[1, 1].set_title('Character Length Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dataset analysis saved to {save_path}")
    
    plt.show()
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"  Total samples: {len(labels)}")
    print(f"  Total words: {total_words}")
    print(f"  Unique words: {unique_words}")
    print(f"  Vocabulary coverage: {unique_words/total_words*100:.2f}%")
    print(f"  Avg words per label: {np.mean(label_lengths):.2f}")
    print(f"  Avg characters per label: {np.mean(char_lengths):.2f}")
    print(f"  Min words: {min(label_lengths)}")
    print(f"  Max words: {max(label_lengths)}")


# Example usage
if __name__ == "__main__":
    # Test visualization functions
    import json
    
    # Example: Plot training curves
    # Giả sử có file training_history.json
    try:
        with open('checkpoints/training_history.json', 'r') as f:
            history = json.load(f)
        
        plot_training_curves(history, save_path='training_curves.png')
    except FileNotFoundError:
        print("No training history found")
    
    # Example: Analyze dataset
    # Giả sử có meta.csv
    try:
        import pandas as pd
        df = pd.read_csv('meta.csv')
        labels = df['label_vi'].tolist()
        
        analyze_dataset_distribution(labels, save_path='dataset_analysis.png')
    except FileNotFoundError:
        print("No meta.csv found")

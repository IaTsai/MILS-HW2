"""
Visualization utilities for multi-task learning model outputs.
Provides visualization for detection, segmentation, and classification results.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Dict, List, Tuple, Optional, Union
import os


class TaskVisualizer:
    """Visualize outputs from multi-task learning model."""
    
    def __init__(self):
        # Color palette for segmentation (PASCAL VOC colors)
        self.seg_colors = self._get_voc_palette()
        
        # Color palette for detection (COCO-style)
        self.det_colors = self._get_detection_colors()
        
        # Class names
        self.voc_classes = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
            'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
            'train', 'tvmonitor'
        ]
        
        self.imagenette_classes = [
            'tench', 'English springer', 'cassette player', 'chain saw',
            'church', 'French horn', 'garbage truck', 'gas pump',
            'golf ball', 'parachute'
        ]
        
        self.coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light'
        ]
    
    def _get_voc_palette(self):
        """Get PASCAL VOC color palette."""
        palette = np.array([
            [0, 0, 0],        # background
            [128, 0, 0],      # aeroplane
            [0, 128, 0],      # bicycle
            [128, 128, 0],    # bird
            [0, 0, 128],      # boat
            [128, 0, 128],    # bottle
            [0, 128, 128],    # bus
            [128, 128, 128],  # car
            [64, 0, 0],       # cat
            [192, 0, 0],      # chair
            [64, 128, 0],     # cow
            [192, 128, 0],    # diningtable
            [64, 0, 128],     # dog
            [192, 0, 128],    # horse
            [64, 128, 128],   # motorbike
            [192, 128, 128],  # person
            [0, 64, 0],       # pottedplant
            [128, 64, 0],     # sheep
            [0, 192, 0],      # sofa
            [128, 192, 0],    # train
            [0, 64, 128]      # tvmonitor
        ], dtype=np.uint8)
        return palette
    
    def _get_detection_colors(self):
        """Get colors for detection visualization."""
        colors = [
            (255, 0, 0),      # Red
            (0, 255, 0),      # Green
            (0, 0, 255),      # Blue
            (255, 255, 0),    # Yellow
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Cyan
            (255, 128, 0),    # Orange
            (128, 0, 255),    # Purple
            (0, 255, 128),    # Spring Green
            (255, 0, 128),    # Rose
        ]
        return colors
    
    def visualize_detection(self, image: Union[torch.Tensor, np.ndarray],
                          predictions: Dict[str, torch.Tensor],
                          targets: Optional[Dict[str, torch.Tensor]] = None,
                          score_threshold: float = 0.5,
                          save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize detection results.
        
        Args:
            image: Input image (C, H, W) or (H, W, C)
            predictions: Dict with 'boxes', 'labels', 'scores'
            targets: Optional ground truth with same format
            score_threshold: Minimum score to show detections
            save_path: Optional path to save visualization
            
        Returns:
            Visualization as numpy array
        """
        # Convert image to numpy if needed
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] in [1, 3]:
                image = image.permute(1, 2, 0).cpu().numpy()
            else:
                image = image.cpu().numpy()
        
        # Normalize image to [0, 255]
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        
        # Draw predictions
        if 'boxes' in predictions and len(predictions['boxes']) > 0:
            boxes = predictions['boxes'].cpu().numpy()
            labels = predictions['labels'].cpu().numpy()
            scores = predictions['scores'].cpu().numpy() if 'scores' in predictions else np.ones(len(boxes))
            
            for box, label, score in zip(boxes, labels, scores):
                if score >= score_threshold:
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Get color
                    color = self.det_colors[label % len(self.det_colors)]
                    color_norm = tuple(c / 255.0 for c in color)
                    
                    # Draw box
                    rect = Rectangle((x1, y1), width, height,
                                   linewidth=2, edgecolor=color_norm,
                                   facecolor='none')
                    ax.add_patch(rect)
                    
                    # Draw label
                    class_name = self.coco_classes[label] if label < len(self.coco_classes) else f'class_{label}'
                    label_text = f'{class_name}: {score:.2f}'
                    ax.text(x1, y1 - 5, label_text,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=color_norm, alpha=0.7),
                           fontsize=10, color='white', weight='bold')
        
        # Draw targets if provided
        if targets is not None and 'boxes' in targets and len(targets['boxes']) > 0:
            boxes = targets['boxes'].cpu().numpy()
            labels = targets['labels'].cpu().numpy()
            
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                
                # Draw dotted box for ground truth
                rect = Rectangle((x1, y1), width, height,
                               linewidth=2, edgecolor='white',
                               facecolor='none', linestyle='--')
                ax.add_patch(rect)
        
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
        ax.axis('off')
        
        # Save or return
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            # Read back the saved image
            result = cv2.imread(save_path)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            return result
        else:
            # Convert to numpy array
            fig.canvas.draw()
            result = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            result = result.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return result
    
    def visualize_segmentation(self, image: Union[torch.Tensor, np.ndarray],
                             prediction: torch.Tensor,
                             target: Optional[torch.Tensor] = None,
                             alpha: float = 0.6,
                             save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize segmentation results.
        
        Args:
            image: Input image (C, H, W) or (H, W, C)
            prediction: Predicted segmentation mask (H, W) or (C, H, W)
            target: Optional ground truth mask
            alpha: Transparency for overlay
            save_path: Optional path to save visualization
            
        Returns:
            Visualization as numpy array
        """
        # Convert image to numpy if needed
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] in [1, 3]:
                image = image.permute(1, 2, 0).cpu().numpy()
            else:
                image = image.cpu().numpy()
        
        # Normalize image to [0, 255]
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # Process prediction
        if isinstance(prediction, torch.Tensor):
            if prediction.dim() == 3:  # (C, H, W)
                prediction = torch.argmax(prediction, dim=0)
            prediction = prediction.cpu().numpy()
        
        # Create color mask
        color_mask = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
        for cls_id in range(len(self.seg_colors)):
            mask = prediction == cls_id
            if mask.any():
                color_mask[mask] = self.seg_colors[cls_id]
        
        # Create visualization
        if target is None:
            # Single prediction
            fig, ax = plt.subplots(1, 2, figsize=(16, 8))
            
            # Original image
            ax[0].imshow(image)
            ax[0].set_title('Original Image', fontsize=14)
            ax[0].axis('off')
            
            # Overlay
            overlay = image.copy()
            overlay = cv2.addWeighted(overlay, 1 - alpha, color_mask, alpha, 0)
            ax[1].imshow(overlay)
            ax[1].set_title('Segmentation Result', fontsize=14)
            ax[1].axis('off')
        else:
            # Prediction vs target
            if isinstance(target, torch.Tensor):
                target = target.cpu().numpy()
            
            # Create target color mask
            target_color_mask = np.zeros((target.shape[0], target.shape[1], 3), dtype=np.uint8)
            for cls_id in range(len(self.seg_colors)):
                mask = target == cls_id
                if mask.any():
                    target_color_mask[mask] = self.seg_colors[cls_id]
            
            fig, ax = plt.subplots(1, 3, figsize=(20, 8))
            
            # Original image
            ax[0].imshow(image)
            ax[0].set_title('Original Image', fontsize=14)
            ax[0].axis('off')
            
            # Prediction overlay
            pred_overlay = image.copy()
            pred_overlay = cv2.addWeighted(pred_overlay, 1 - alpha, color_mask, alpha, 0)
            ax[1].imshow(pred_overlay)
            ax[1].set_title('Prediction', fontsize=14)
            ax[1].axis('off')
            
            # Target overlay
            target_overlay = image.copy()
            target_overlay = cv2.addWeighted(target_overlay, 1 - alpha, target_color_mask, alpha, 0)
            ax[2].imshow(target_overlay)
            ax[2].set_title('Ground Truth', fontsize=14)
            ax[2].axis('off')
        
        # Add legend
        self._add_segmentation_legend(fig)
        
        # Save or return
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            # Read back the saved image
            result = cv2.imread(save_path)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            return result
        else:
            # Convert to numpy array
            fig.canvas.draw()
            result = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            result = result.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return result
    
    def _add_segmentation_legend(self, fig):
        """Add legend for segmentation classes."""
        # Create legend elements
        from matplotlib.patches import Patch
        legend_elements = []
        
        # Only include classes that might be present
        for i in range(min(10, len(self.voc_classes))):  # Show first 10 classes
            color = self.seg_colors[i] / 255.0
            legend_elements.append(Patch(facecolor=color, label=self.voc_classes[i]))
        
        # Add legend to figure
        fig.legend(handles=legend_elements, loc='lower center', 
                  ncol=5, frameon=True, fancybox=True, shadow=True)
    
    def visualize_classification(self, image: Union[torch.Tensor, np.ndarray],
                               prediction: torch.Tensor,
                               target: Optional[int] = None,
                               top_k: int = 5,
                               save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize classification results.
        
        Args:
            image: Input image (C, H, W) or (H, W, C)
            prediction: Class probabilities (num_classes,)
            target: Optional ground truth label
            top_k: Number of top predictions to show
            save_path: Optional path to save visualization
            
        Returns:
            Visualization as numpy array
        """
        # Convert image to numpy if needed
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] in [1, 3]:
                image = image.permute(1, 2, 0).cpu().numpy()
            else:
                image = image.cpu().numpy()
        
        # Normalize image to [0, 255]
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # Get predictions
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.cpu()
        
        # Apply softmax if needed
        if prediction.dim() == 1:
            probs = torch.softmax(prediction, dim=0)
        else:
            probs = prediction
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))
        
        # Create visualization
        fig = plt.figure(figsize=(12, 8))
        
        # Image subplot
        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(image)
        
        # Add title with prediction
        pred_class = top_indices[0].item()
        pred_prob = top_probs[0].item()
        class_name = self.imagenette_classes[pred_class] if pred_class < len(self.imagenette_classes) else f'class_{pred_class}'
        
        title = f'Predicted: {class_name} ({pred_prob:.2%})'
        if target is not None:
            target_name = self.imagenette_classes[target] if target < len(self.imagenette_classes) else f'class_{target}'
            title += f'\nActual: {target_name}'
            if pred_class == target:
                title_color = 'green'
            else:
                title_color = 'red'
        else:
            title_color = 'black'
        
        ax1.set_title(title, fontsize=14, color=title_color, weight='bold')
        ax1.axis('off')
        
        # Probability bar chart
        ax2 = plt.subplot(1, 2, 2)
        
        # Prepare data for bar chart
        y_labels = []
        x_values = []
        colors = []
        
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            idx = idx.item()
            class_name = self.imagenette_classes[idx] if idx < len(self.imagenette_classes) else f'class_{idx}'
            y_labels.append(class_name)
            x_values.append(prob.item())
            
            if target is not None and idx == target:
                colors.append('green')
            elif i == 0:
                colors.append('darkblue')
            else:
                colors.append('lightblue')
        
        # Create horizontal bar chart
        y_pos = np.arange(len(y_labels))
        ax2.barh(y_pos, x_values, color=colors, alpha=0.8)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(y_labels)
        ax2.set_xlabel('Probability', fontsize=12)
        ax2.set_title('Top-{} Predictions'.format(top_k), fontsize=14)
        ax2.set_xlim(0, 1)
        
        # Add probability values on bars
        for i, (prob, label) in enumerate(zip(x_values, y_labels)):
            ax2.text(prob + 0.01, i, f'{prob:.1%}', 
                    va='center', fontsize=10, weight='bold')
        
        plt.tight_layout()
        
        # Save or return
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            # Read back the saved image
            result = cv2.imread(save_path)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            return result
        else:
            # Convert to numpy array
            fig.canvas.draw()
            result = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            result = result.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return result
    
    def visualize_multi_task(self, image: Union[torch.Tensor, np.ndarray],
                           outputs: Dict[str, torch.Tensor],
                           targets: Optional[Dict[str, torch.Tensor]] = None,
                           save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize all three tasks in one figure.
        
        Args:
            image: Input image
            outputs: Dict with 'detection', 'segmentation', 'classification' outputs
            targets: Optional ground truth for all tasks
            save_path: Optional path to save visualization
            
        Returns:
            Combined visualization as numpy array
        """
        # Convert image to numpy if needed
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] in [1, 3]:
                image_np = image.permute(1, 2, 0).cpu().numpy()
            else:
                image_np = image.cpu().numpy()
        else:
            image_np = image
        
        # Normalize image to [0, 255]
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # Original image
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(image_np)
        ax1.set_title('Original Image', fontsize=16, weight='bold')
        ax1.axis('off')
        
        # Detection visualization
        ax2 = plt.subplot(2, 3, 2)
        if 'detection' in outputs:
            det_vis = self._visualize_detection_simple(image_np, outputs['detection'])
            ax2.imshow(det_vis)
            ax2.set_title('Object Detection', fontsize=16, weight='bold')
        else:
            ax2.imshow(image_np)
            ax2.set_title('Object Detection (N/A)', fontsize=16)
        ax2.axis('off')
        
        # Segmentation visualization
        ax3 = plt.subplot(2, 3, 3)
        if 'segmentation' in outputs:
            seg_vis = self._visualize_segmentation_simple(image_np, outputs['segmentation'])
            ax3.imshow(seg_vis)
            ax3.set_title('Semantic Segmentation', fontsize=16, weight='bold')
        else:
            ax3.imshow(image_np)
            ax3.set_title('Semantic Segmentation (N/A)', fontsize=16)
        ax3.axis('off')
        
        # Classification results
        ax4 = plt.subplot(2, 3, 4)
        ax4.axis('off')  # We'll use text instead
        
        if 'classification' in outputs:
            cls_output = outputs['classification']
            if isinstance(cls_output, torch.Tensor):
                if cls_output.dim() == 2:
                    cls_output = cls_output[0]  # Take first in batch
                probs = torch.softmax(cls_output, dim=0)
                top_probs, top_indices = torch.topk(probs, 3)
                
                # Display top-3 predictions as text
                text = "Classification Results:\n\n"
                for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                    idx = idx.item()
                    class_name = self.imagenette_classes[idx] if idx < len(self.imagenette_classes) else f'class_{idx}'
                    text += f"{i+1}. {class_name}: {prob:.1%}\n"
                
                ax4.text(0.1, 0.5, text, fontsize=14, transform=ax4.transAxes,
                        verticalalignment='center', family='monospace',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
            ax4.set_title('Image Classification', fontsize=16, weight='bold')
        
        # Performance metrics
        ax5 = plt.subplot(2, 3, 5)
        ax5.axis('off')
        
        # Display some metrics if available
        metrics_text = "Performance Metrics:\n\n"
        metrics_text += "• Detection: mAP evaluation needed\n"
        metrics_text += "• Segmentation: mIoU evaluation needed\n"
        metrics_text += "• Classification: Top-1 accuracy needed\n"
        
        ax5.text(0.1, 0.5, metrics_text, fontsize=12, transform=ax5.transAxes,
                verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        ax5.set_title('Metrics Summary', fontsize=16, weight='bold')
        
        # Model info
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        info_text = "Model Architecture:\n\n"
        info_text += "• Backbone: MobileNetV3-Small\n"
        info_text += "• Neck: FPN\n"
        info_text += "• Heads: Independent task heads\n"
        info_text += "• Training: Joint training strategy\n"
        
        ax6.text(0.1, 0.5, info_text, fontsize=12, transform=ax6.transAxes,
                verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        ax6.set_title('Model Information', fontsize=16, weight='bold')
        
        plt.suptitle('Multi-Task Learning Results', fontsize=20, weight='bold')
        plt.tight_layout()
        
        # Save or return
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            # Read back the saved image
            result = cv2.imread(save_path)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            return result
        else:
            # Convert to numpy array
            fig.canvas.draw()
            result = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            result = result.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return result
    
    def _visualize_detection_simple(self, image, det_output):
        """Simple detection visualization for multi-task view."""
        vis_image = image.copy()
        
        # Parse detection output
        if isinstance(det_output, torch.Tensor) and det_output.dim() == 2:
            # Simple threshold-based visualization
            # This is a placeholder - should use proper detection parsing
            pass
        
        return vis_image
    
    def _visualize_segmentation_simple(self, image, seg_output):
        """Simple segmentation visualization for multi-task view."""
        if isinstance(seg_output, torch.Tensor):
            if seg_output.dim() == 3:  # (C, H, W)
                seg_mask = torch.argmax(seg_output, dim=0).cpu().numpy()
            else:
                seg_mask = seg_output.cpu().numpy()
            
            # Create color mask
            color_mask = np.zeros((seg_mask.shape[0], seg_mask.shape[1], 3), dtype=np.uint8)
            for cls_id in range(min(len(self.seg_colors), seg_mask.max() + 1)):
                mask = seg_mask == cls_id
                if mask.any():
                    color_mask[mask] = self.seg_colors[cls_id]
            
            # Overlay
            overlay = cv2.addWeighted(image, 0.6, color_mask, 0.4, 0)
            return overlay
        
        return image


# Legacy functions for compatibility
def visualize_detection(image, boxes, labels, scores=None, class_names=None, threshold=0.5):
    fig, ax = plt.subplots(1, figsize=(12, 8))
    
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
        if image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
    
    ax.imshow(image)
    
    for i, (box, label) in enumerate(zip(boxes, labels)):
        if scores is not None and scores[i] < threshold:
            continue
        
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, 
                                edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        label_text = f'{label}'
        if class_names:
            label_text = class_names[label]
        if scores is not None:
            label_text += f' {scores[i]:.2f}'
        
        ax.text(x1, y1 - 5, label_text, color='white', 
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.5))
    
    ax.axis('off')
    plt.tight_layout()
    return fig


def visualize_segmentation(image, mask, num_classes=21, alpha=0.6):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
        if image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
    
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    cmap = plt.get_cmap('tab20')
    colors = cmap(np.linspace(0, 1, num_classes))
    
    colored_mask = np.zeros((*mask.shape, 3))
    for c in range(num_classes):
        colored_mask[mask == c] = colors[c][:3]
    
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(colored_mask)
    ax2.set_title('Segmentation Mask')
    ax2.axis('off')
    
    overlay = image * (1 - alpha) + colored_mask * alpha
    ax3.imshow(overlay)
    ax3.set_title('Overlay')
    ax3.axis('off')
    
    plt.tight_layout()
    return fig


def plot_training_curves(history):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    epochs = range(1, len(history['total_loss']) + 1)
    
    axes[0, 0].plot(epochs, history['total_loss'], 'b-', label='Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(epochs, history['det_loss'], 'r-', label='Detection Loss')
    axes[0, 1].set_title('Detection Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[0, 2].plot(epochs, history['seg_loss'], 'g-', label='Segmentation Loss')
    axes[0, 2].set_title('Segmentation Loss')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    axes[1, 0].plot(epochs, history['cls_loss'], 'm-', label='Classification Loss')
    axes[1, 0].set_title('Classification Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    if 'seg_miou' in history:
        axes[1, 1].plot(epochs, history['seg_miou'], 'g-', label='Segmentation mIoU')
        axes[1, 1].set_title('Segmentation mIoU')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('mIoU')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    if 'cls_acc' in history:
        axes[1, 2].plot(epochs, history['cls_acc'], 'm-', label='Classification Accuracy')
        axes[1, 2].set_title('Classification Accuracy')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Accuracy')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
    
    plt.tight_layout()
    return fig
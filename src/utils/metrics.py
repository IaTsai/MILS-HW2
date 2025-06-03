"""
Metrics calculation utilities for multi-task learning evaluation.
Includes mAP for detection, mIoU for segmentation, and accuracy for classification.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import time
from collections import defaultdict


class MetricsCalculator:
    """Calculate metrics for multi-task learning evaluation."""
    
    def __init__(self):
        self.eps = 1e-6
        
    def calculate_map(self, predictions: List[Dict], targets: List[Dict], 
                     iou_threshold: float = 0.5, 
                     class_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Calculate mean Average Precision (mAP) for object detection.
        
        Args:
            predictions: List of dicts with 'boxes', 'labels', 'scores'
            targets: List of dicts with 'boxes', 'labels'
            iou_threshold: IoU threshold for matching
            class_names: Optional class names for per-class AP
            
        Returns:
            Dict with 'mAP' and per-class AP if class_names provided
        """
        if not predictions or not targets:
            return {'mAP': 0.0}
            
        # Collect all unique classes
        all_classes = set()
        for pred in predictions:
            if 'labels' in pred and len(pred['labels']) > 0:
                all_classes.update(pred['labels'].tolist() if torch.is_tensor(pred['labels']) else pred['labels'])
        for target in targets:
            if 'labels' in target and len(target['labels']) > 0:
                all_classes.update(target['labels'].tolist() if torch.is_tensor(target['labels']) else target['labels'])
        
        if not all_classes:
            return {'mAP': 0.0}
            
        # Calculate AP for each class
        ap_per_class = {}
        for class_id in all_classes:
            ap = self._calculate_ap_for_class(predictions, targets, class_id, iou_threshold)
            ap_per_class[class_id] = ap
            
        # Calculate mAP
        mAP = np.mean(list(ap_per_class.values())) if ap_per_class else 0.0
        
        result = {'mAP': float(mAP)}
        
        # Add per-class AP if class names provided
        if class_names:
            for class_id, ap in ap_per_class.items():
                if class_id < len(class_names):
                    result[f'AP_{class_names[class_id]}'] = float(ap)
                    
        return result
    
    def _calculate_ap_for_class(self, predictions: List[Dict], targets: List[Dict], 
                               class_id: int, iou_threshold: float) -> float:
        """Calculate Average Precision for a specific class."""
        # Collect all predictions and targets for this class
        all_pred_boxes = []
        all_pred_scores = []
        all_target_boxes = []
        
        for img_idx, (pred, target) in enumerate(zip(predictions, targets)):
            # Get predictions for this class
            if 'labels' in pred and len(pred['labels']) > 0:
                pred_labels = pred['labels'].cpu() if torch.is_tensor(pred['labels']) else torch.tensor(pred['labels'])
                pred_mask = pred_labels == class_id
                if pred_mask.any():
                    pred_boxes = pred['boxes'][pred_mask].cpu() if torch.is_tensor(pred['boxes']) else torch.tensor(pred['boxes'][pred_mask])
                    pred_scores = pred['scores'][pred_mask].cpu() if torch.is_tensor(pred['scores']) else torch.tensor(pred['scores'][pred_mask])
                    
                    for box, score in zip(pred_boxes, pred_scores):
                        all_pred_boxes.append((img_idx, box))
                        all_pred_scores.append(score)
            
            # Get targets for this class
            if 'labels' in target and len(target['labels']) > 0:
                target_labels = target['labels'].cpu() if torch.is_tensor(target['labels']) else torch.tensor(target['labels'])
                target_mask = target_labels == class_id
                if target_mask.any():
                    target_boxes = target['boxes'][target_mask].cpu() if torch.is_tensor(target['boxes']) else torch.tensor(target['boxes'][target_mask])
                    for box in target_boxes:
                        all_target_boxes.append((img_idx, box))
        
        if not all_pred_boxes or not all_target_boxes:
            return 0.0
            
        # Sort predictions by score
        sorted_indices = np.argsort(all_pred_scores)[::-1]
        all_pred_boxes = [all_pred_boxes[i] for i in sorted_indices]
        
        # Track which targets have been matched
        matched_targets = set()
        
        # Calculate precision and recall
        tp = 0
        fp = 0
        num_targets = len(all_target_boxes)
        
        precisions = []
        recalls = []
        
        for pred_idx, (pred_img_idx, pred_box) in enumerate(all_pred_boxes):
            # Find best matching target
            best_iou = 0
            best_target_idx = -1
            
            for target_idx, (target_img_idx, target_box) in enumerate(all_target_boxes):
                if pred_img_idx == target_img_idx and target_idx not in matched_targets:
                    iou = self._calculate_iou(pred_box, target_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_target_idx = target_idx
            
            # Check if it's a true positive
            if best_iou >= iou_threshold and best_target_idx >= 0:
                tp += 1
                matched_targets.add(best_target_idx)
            else:
                fp += 1
                
            # Calculate precision and recall
            precision = tp / (tp + fp + self.eps)
            recall = tp / (num_targets + self.eps)
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Calculate AP using 11-point interpolation
        ap = 0
        for t in np.linspace(0, 1, 11):
            if np.sum(np.array(recalls) >= t) == 0:
                p = 0
            else:
                p = np.max(np.array(precisions)[np.array(recalls) >= t])
            ap += p / 11
            
        return ap
    
    def _calculate_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> float:
        """Calculate IoU between two boxes."""
        # Convert to numpy if needed
        if torch.is_tensor(box1):
            box1 = box1.numpy()
        if torch.is_tensor(box2):
            box2 = box2.numpy()
            
        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + self.eps)
    
    def calculate_miou(self, predictions: torch.Tensor, targets: torch.Tensor, 
                      num_classes: int = 21, ignore_index: int = 255) -> Dict[str, float]:
        """
        Calculate mean Intersection over Union (mIoU) for semantic segmentation.
        
        Args:
            predictions: (B, H, W) or (B, C, H, W) tensor of predictions
            targets: (B, H, W) tensor of ground truth labels
            num_classes: Number of classes
            ignore_index: Label to ignore
            
        Returns:
            Dict with 'mIoU' and per-class IoU
        """
        # Handle different input formats
        if predictions.dim() == 4:  # (B, C, H, W)
            predictions = torch.argmax(predictions, dim=1)
        
        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Remove ignored pixels
        mask = targets != ignore_index
        predictions = predictions[mask]
        targets = targets[mask]
        
        # Calculate IoU for each class
        ious = []
        per_class_iou = {}
        
        for cls in range(num_classes):
            pred_mask = predictions == cls
            target_mask = targets == cls
            
            intersection = (pred_mask & target_mask).sum().float()
            union = (pred_mask | target_mask).sum().float()
            
            if union > 0:
                iou = intersection / union
                ious.append(iou.item())
                per_class_iou[f'IoU_class_{cls}'] = iou.item()
            else:
                # Skip classes not present in targets
                per_class_iou[f'IoU_class_{cls}'] = float('nan')
        
        # Calculate mIoU (only for classes present in targets)
        miou = np.nanmean(ious) if ious else 0.0
        
        result = {'mIoU': float(miou)}
        result.update(per_class_iou)
        
        return result
    
    def calculate_top1_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Calculate Top-1 accuracy for classification.
        
        Args:
            predictions: (B, num_classes) tensor of logits or probabilities
            targets: (B,) tensor of ground truth labels
            
        Returns:
            Dict with 'top1_accuracy' and additional metrics
        """
        if predictions.size(0) == 0:
            return {'top1_accuracy': 0.0}
            
        # Get predicted classes
        _, predicted = torch.max(predictions, 1)
        
        # Calculate accuracy
        correct = (predicted == targets).sum().item()
        total = targets.size(0)
        accuracy = correct / total if total > 0 else 0.0
        
        # Calculate per-class accuracy
        per_class_correct = defaultdict(int)
        per_class_total = defaultdict(int)
        
        for pred, target in zip(predicted, targets):
            per_class_total[target.item()] += 1
            if pred == target:
                per_class_correct[target.item()] += 1
        
        per_class_acc = {}
        for cls in per_class_total:
            acc = per_class_correct[cls] / per_class_total[cls]
            per_class_acc[f'accuracy_class_{cls}'] = acc
        
        result = {
            'top1_accuracy': float(accuracy),
            'correct_predictions': correct,
            'total_predictions': total
        }
        result.update(per_class_acc)
        
        return result
    
    def calculate_inference_time(self, model: torch.nn.Module, 
                               input_size: Tuple[int, int, int] = (3, 512, 512),
                               batch_size: int = 1,
                               num_runs: int = 100,
                               warmup_runs: int = 10,
                               device: str = 'cuda') -> Dict[str, float]:
        """
        Calculate inference time for the model.
        
        Args:
            model: PyTorch model to evaluate
            input_size: Input tensor size (C, H, W)
            batch_size: Batch size for inference
            num_runs: Number of runs for timing
            warmup_runs: Number of warmup runs
            device: Device to run on
            
        Returns:
            Dict with timing statistics
        """
        model.eval()
        model = model.to(device)
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, *input_size).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(dummy_input)
        
        # Synchronize if using CUDA
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # Time inference
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                if device == 'cuda':
                    torch.cuda.synchronize()
                    
                start_time = time.time()
                _ = model(dummy_input)
                
                if device == 'cuda':
                    torch.cuda.synchronize()
                    
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        times = np.array(times)
        
        # Calculate statistics
        result = {
            'mean_inference_time_ms': float(np.mean(times)),
            'std_inference_time_ms': float(np.std(times)),
            'min_inference_time_ms': float(np.min(times)),
            'max_inference_time_ms': float(np.max(times)),
            'median_inference_time_ms': float(np.median(times)),
            'batch_size': batch_size,
            'input_size': input_size,
            'num_runs': num_runs
        }
        
        return result
    
    def calculate_model_size(self, model: torch.nn.Module) -> Dict[str, Union[int, float]]:
        """
        Calculate model size and parameter count.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dict with model size information
        """
        total_params = 0
        trainable_params = 0
        
        for param in model.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        # Calculate model size in MB
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
            
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        model_size_mb = (param_size + buffer_size) / 1024 / 1024
        
        result = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'model_size_mb': float(model_size_mb),
            'parameter_size_mb': float(param_size / 1024 / 1024),
            'buffer_size_mb': float(buffer_size / 1024 / 1024)
        }
        
        return result
    
    def calculate_memory_usage(self, model: torch.nn.Module, 
                             input_size: Tuple[int, int, int] = (3, 512, 512),
                             batch_size: int = 1,
                             device: str = 'cuda') -> Dict[str, float]:
        """
        Estimate memory usage during inference.
        
        Args:
            model: PyTorch model
            input_size: Input tensor size
            batch_size: Batch size
            device: Device to run on
            
        Returns:
            Dict with memory usage information
        """
        if device != 'cuda' or not torch.cuda.is_available():
            return {'error': 'CUDA not available for memory measurement'}
        
        model.eval()
        model = model.to(device)
        
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Initial memory
        initial_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        # Create input and run inference
        dummy_input = torch.randn(batch_size, *input_size).to(device)
        
        with torch.no_grad():
            _ = model(dummy_input)
        
        # Peak memory
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        current_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        result = {
            'initial_memory_mb': float(initial_memory),
            'peak_memory_mb': float(peak_memory),
            'current_memory_mb': float(current_memory),
            'inference_memory_mb': float(peak_memory - initial_memory),
            'batch_size': batch_size,
            'input_size': input_size
        }
        
        return result


# Legacy classes for compatibility
class DetectionMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.tp = np.zeros(self.num_classes)
        self.fp = np.zeros(self.num_classes)
        self.fn = np.zeros(self.num_classes)
    
    def update(self, pred_boxes, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5):
        pass
    
    def compute_map(self):
        precisions = []
        recalls = []
        
        for c in range(self.num_classes):
            if self.tp[c] + self.fp[c] == 0:
                precision = 0
            else:
                precision = self.tp[c] / (self.tp[c] + self.fp[c])
            
            if self.tp[c] + self.fn[c] == 0:
                recall = 0
            else:
                recall = self.tp[c] / (self.tp[c] + self.fn[c])
            
            precisions.append(precision)
            recalls.append(recall)
        
        return np.mean(precisions), precisions, recalls


class SegmentationMetrics:
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, pred, target):
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        
        mask = (target != self.ignore_index)
        pred = pred[mask]
        target = target[mask]
        
        for t, p in zip(target.flatten(), pred.flatten()):
            self.confusion_matrix[t, p] += 1
    
    def compute_miou(self):
        iou_per_class = []
        
        for c in range(self.num_classes):
            intersection = self.confusion_matrix[c, c]
            union = (self.confusion_matrix[c, :].sum() + 
                    self.confusion_matrix[:, c].sum() - intersection)
            
            if union == 0:
                iou = 0
            else:
                iou = intersection / union
            
            iou_per_class.append(iou)
        
        miou = np.mean(iou_per_class)
        return miou, iou_per_class
    
    def compute_pixel_accuracy(self):
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        return correct / total if total > 0 else 0


class ClassificationMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.targets = []
    
    def update(self, pred, target):
        _, pred_classes = pred.max(1)
        self.predictions.extend(pred_classes.cpu().numpy())
        self.targets.extend(target.cpu().numpy())
    
    def compute_accuracy(self):
        from sklearn.metrics import accuracy_score
        return accuracy_score(self.targets, self.predictions)
    
    def compute_precision_recall_f1(self):
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.targets, self.predictions, average='macro'
        )
        return precision, recall, f1
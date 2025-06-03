#!/usr/bin/env python
"""
Comprehensive evaluation script for multi-task learning model.
Evaluates detection (mAP), segmentation (mIoU), and classification (Top-1 accuracy).
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.metrics import MetricsCalculator
from src.datasets.unified_dataloader import create_unified_dataloaders
from src.datasets.coco_dataset import CocoDetectionDataset
from src.datasets.voc_dataset import VOCSegmentationDataset
from src.datasets.imagenette_dataset import ImagenetteDataset


def load_model(weights_path: str, device: str = 'cuda'):
    """Load trained model from checkpoint."""
    print(f"Loading model from {weights_path}...")
    
    # Check if file exists
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    
    # Load checkpoint
    checkpoint = torch.load(weights_path, map_location=device)
    
    # Determine model type and load accordingly
    if 'model_state_dict' in checkpoint:
        # Standard checkpoint format
        state_dict = checkpoint['model_state_dict']
        
        # Try to load the improved model first
        try:
            from improved_model import ImprovedMultiTaskModel
            model = ImprovedMultiTaskModel()
            model.load_state_dict(state_dict)
            print("Loaded ImprovedUnifiedModel")
        except:
            # Fall back to original unified model
            from src.models.unified_model import UnifiedMultiTaskModel
            model = UnifiedMultiTaskModel(
                num_det_classes=10,
                num_seg_classes=21,
                num_cls_classes=10
            )
            model.load_state_dict(state_dict)
            print("Loaded UnifiedMultiTaskModel")
    else:
        # Direct state dict
        try:
            from improved_model import ImprovedMultiTaskModel
            model = ImprovedMultiTaskModel()
            model.load_state_dict(checkpoint)
            print("Loaded ImprovedUnifiedModel from state dict")
        except:
            from src.models.unified_model import UnifiedMultiTaskModel
            model = UnifiedMultiTaskModel(
                num_det_classes=10,
                num_seg_classes=21,
                num_cls_classes=10
            )
            model.load_state_dict(checkpoint)
            print("Loaded UnifiedMultiTaskModel from state dict")
    
    model = model.to(device)
    model.eval()
    
    return model


def evaluate_detection(model, dataloader, device='cuda', num_classes=10):
    """Evaluate detection performance."""
    print("\nEvaluating detection task...")
    
    metrics_calc = MetricsCalculator()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Detection evaluation")):
            # Handle different batch formats
            if isinstance(batch, dict):
                images = batch['image'].to(device)
                targets = batch.get('detection', None)
            elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                images = batch[0].to(device)
                if isinstance(batch[1], dict):
                    targets = batch[1].get('boxes', None)
                else:
                    targets = None
            else:
                continue
            
            # Get predictions
            outputs = model(images)
            
            # Process detection outputs
            if 'detection' in outputs:
                det_outputs = outputs['detection']
                batch_size = images.size(0)
                
                for i in range(batch_size):
                    # Parse detection predictions
                    pred_dict = parse_detection_output(det_outputs[i], score_threshold=0.05)
                    all_predictions.append(pred_dict)
                    
                    # Parse targets if available
                    if targets is not None:
                        if isinstance(targets, dict) and 'boxes' in targets:
                            target_dict = {
                                'boxes': targets['boxes'][i].cpu(),
                                'labels': targets['labels'][i].cpu()
                            }
                        else:
                            target_dict = {'boxes': torch.empty(0, 4), 'labels': torch.empty(0)}
                        all_targets.append(target_dict)
    
    # Calculate mAP
    if all_predictions and all_targets:
        detection_metrics = metrics_calc.calculate_map(all_predictions, all_targets)
        return detection_metrics
    else:
        return {'mAP': 0.0}


def parse_detection_output(det_output, score_threshold=0.05):
    """Parse detection output into standard format."""
    # det_output shape: [H*W, 15] where 15 = 4 (box) + 10 (classes) + 1 (centerness)
    
    if det_output.dim() == 1:
        det_output = det_output.unsqueeze(0)
    
    # Extract components
    boxes = det_output[:, :4]  # [N, 4]
    class_scores = det_output[:, 4:14]  # [N, 10]
    centerness = det_output[:, 14:15]  # [N, 1]
    
    # Apply sigmoid to get probabilities
    boxes = torch.sigmoid(boxes)
    class_probs = torch.sigmoid(class_scores)
    centerness_prob = torch.sigmoid(centerness)
    
    # Get best class and score for each location
    max_scores, max_classes = class_probs.max(dim=1)
    
    # Combine with centerness
    final_scores = max_scores * centerness_prob.squeeze(-1)
    
    # Filter by threshold
    keep = final_scores > score_threshold
    
    if keep.sum() == 0:
        return {
            'boxes': torch.empty(0, 4),
            'labels': torch.empty(0, dtype=torch.long),
            'scores': torch.empty(0)
        }
    
    # Convert normalized boxes to absolute coordinates
    # Assuming boxes are in format [x_center, y_center, width, height] normalized
    filtered_boxes = boxes[keep]
    x_center = filtered_boxes[:, 0] * 512
    y_center = filtered_boxes[:, 1] * 512
    width = filtered_boxes[:, 2] * 512
    height = filtered_boxes[:, 3] * 512
    
    # Convert to [x1, y1, x2, y2]
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    final_boxes = torch.stack([x1, y1, x2, y2], dim=1)
    
    return {
        'boxes': final_boxes.cpu(),
        'labels': max_classes[keep].cpu(),
        'scores': final_scores[keep].cpu()
    }


def evaluate_segmentation(model, dataloader, device='cuda', num_classes=21):
    """Evaluate segmentation performance."""
    print("\nEvaluating segmentation task...")
    
    metrics_calc = MetricsCalculator()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Segmentation evaluation")):
            # The datasets return (image, mask) tuples
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                images, masks = batch
                images = images.to(device)
                masks = masks.to(device)
            else:
                continue
                
            # Get predictions
            outputs = model(images)
            
            # Process segmentation outputs
            if 'segmentation' in outputs:
                seg_outputs = outputs['segmentation']
                
                # Resize to match target size if needed
                if seg_outputs.shape[-2:] != masks.shape[-2:]:
                    seg_outputs = F.interpolate(
                        seg_outputs, 
                        size=masks.shape[-2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                all_predictions.append(seg_outputs.cpu())
                all_targets.append(masks.cpu())
    
    # Calculate mIoU
    if all_predictions and all_targets:
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        segmentation_metrics = metrics_calc.calculate_miou(all_predictions, all_targets, num_classes=num_classes)
        return segmentation_metrics
    else:
        return {'mIoU': 0.0}


def evaluate_classification(model, dataloader, device='cuda', num_classes=10):
    """Evaluate classification performance."""
    print("\nEvaluating classification task...")
    
    metrics_calc = MetricsCalculator()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Classification evaluation")):
            # The datasets return (image, target) tuples where target might be dict or tensor
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                images, labels = batch
                images = images.to(device)
                
                # Handle different label formats
                if isinstance(labels, dict):
                    if 'labels' in labels:
                        labels = labels['labels'].to(device)
                    elif 'classification' in labels:
                        labels = labels['classification'].to(device)
                    elif 'label' in labels:
                        labels = labels['label'].to(device)
                    else:
                        continue
                else:
                    labels = labels.to(device)
            else:
                continue
                
            # Get predictions
            outputs = model(images)
            
            # Process classification outputs
            if 'classification' in outputs:
                cls_outputs = outputs['classification']
                all_predictions.append(cls_outputs.cpu())
                all_targets.append(labels.cpu())
    
    # Calculate accuracy
    if all_predictions and all_targets:
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        classification_metrics = metrics_calc.calculate_top1_accuracy(all_predictions, all_targets)
        return classification_metrics
    else:
        return {'top1_accuracy': 0.0}


def evaluate_all_tasks(model, data_root, batch_size=8, device='cuda'):
    """Evaluate all three tasks."""
    results = {}
    
    # Create task-specific dataloaders
    print("\nCreating evaluation dataloaders...")
    
    # Detection dataloader
    det_dataset = CocoDetectionDataset(
        root_dir=os.path.join(data_root, 'mini_coco_det'),
        split='val'
    )
    det_loader = torch.utils.data.DataLoader(
        det_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        collate_fn=CocoDetectionDataset.collate_fn
    )
    
    # Segmentation dataloader
    seg_dataset = VOCSegmentationDataset(
        root_dir=os.path.join(data_root, 'mini_voc_seg'),
        split='val'
    )
    seg_loader = torch.utils.data.DataLoader(
        seg_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Classification dataloader
    cls_dataset = ImagenetteDataset(
        root_dir=os.path.join(data_root, 'imagenette_160'),
        split='val'
    )
    cls_loader = torch.utils.data.DataLoader(
        cls_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Evaluate each task
    results['detection'] = evaluate_detection(model, det_loader, device)
    results['segmentation'] = evaluate_segmentation(model, seg_loader, device)
    results['classification'] = evaluate_classification(model, cls_loader, device)
    
    return results


def evaluate_efficiency(model, device='cuda'):
    """Evaluate model efficiency metrics."""
    print("\nEvaluating model efficiency...")
    
    metrics_calc = MetricsCalculator()
    
    # Model size
    size_metrics = metrics_calc.calculate_model_size(model)
    
    # Inference time
    time_metrics = metrics_calc.calculate_inference_time(model, device=device)
    
    # Memory usage
    memory_metrics = metrics_calc.calculate_memory_usage(model, device=device)
    
    return {
        'model_size': size_metrics,
        'inference_time': time_metrics,
        'memory_usage': memory_metrics
    }


def check_compliance(results):
    """Check if results meet assignment requirements."""
    compliance = {
        'parameter_limit_8m': results['efficiency']['model_size']['total_parameters'] < 8_000_000,
        'inference_speed_150ms': results['efficiency']['inference_time']['mean_inference_time_ms'] < 150,
    }
    
    # Check task-specific thresholds if available
    if 'detection' in results['task_performance']:
        compliance['detection_map_threshold'] = results['task_performance']['detection']['mAP'] >= 0.5
    
    if 'segmentation' in results['task_performance']:
        compliance['segmentation_miou_threshold'] = results['task_performance']['segmentation']['mIoU'] >= 0.7
    
    if 'classification' in results['task_performance']:
        compliance['classification_accuracy_threshold'] = results['task_performance']['classification']['top1_accuracy'] >= 0.7
    
    # Check if all requirements are met
    requirements = [
        compliance['parameter_limit_8m'],
        compliance['inference_speed_150ms']
    ]
    
    if 'detection_map_threshold' in compliance:
        requirements.append(compliance['detection_map_threshold'])
    
    if 'segmentation_miou_threshold' in compliance:
        requirements.append(compliance['segmentation_miou_threshold'])
    
    if 'classification_accuracy_threshold' in compliance:
        requirements.append(compliance['classification_accuracy_threshold'])
    
    compliance['all_requirements_met'] = all(requirements)
    
    return compliance


def generate_report(results, output_dir):
    """Generate evaluation report."""
    report = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'task_performance': results['task_performance'],
        'efficiency_metrics': results['efficiency'],
        'compliance_check': results['compliance'],
        'architecture_analysis': {
            'design_choice': 'independent_task_heads',
            'joint_training_used': True,
            'ewc_used': False
        }
    }
    
    # Save JSON report
    json_path = os.path.join(output_dir, 'metrics_report.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate summary
    summary = f"""
# Evaluation Summary

## Task Performance"""
    
    if 'detection' in results['task_performance']:
        summary += f"\n- **Detection mAP**: {results['task_performance']['detection']['mAP']:.4f}"
    
    if 'segmentation' in results['task_performance']:
        summary += f"\n- **Segmentation mIoU**: {results['task_performance']['segmentation']['mIoU']:.4f}"
    
    if 'classification' in results['task_performance']:
        summary += f"\n- **Classification Top-1**: {results['task_performance']['classification']['top1_accuracy']:.4f}"
    
    summary += f"""

## Efficiency Metrics
- **Total Parameters**: {results['efficiency']['model_size']['total_parameters']:,}
- **Model Size**: {results['efficiency']['model_size']['model_size_mb']:.2f} MB
- **Inference Time**: {results['efficiency']['inference_time']['mean_inference_time_ms']:.2f} ms

## Compliance Check
- **Parameter Limit (<8M)**: {'✅' if results['compliance']['parameter_limit_8m'] else '❌'}
- **Inference Speed (<150ms)**: {'✅' if results['compliance']['inference_speed_150ms'] else '❌'}"""
    
    if 'detection_map_threshold' in results['compliance']:
        summary += f"\n- **Detection mAP (≥50%)**: {'✅' if results['compliance']['detection_map_threshold'] else '❌'}"
    
    if 'segmentation_miou_threshold' in results['compliance']:
        summary += f"\n- **Segmentation mIoU (≥70%)**: {'✅' if results['compliance']['segmentation_miou_threshold'] else '❌'}"
    
    if 'classification_accuracy_threshold' in results['compliance']:
        summary += f"\n- **Classification Acc (≥70%)**: {'✅' if results['compliance']['classification_accuracy_threshold'] else '❌'}"
    
    summary += f"""

**All Requirements Met**: {'✅' if results['compliance']['all_requirements_met'] else '❌'}
"""
    
    # Save summary
    summary_path = os.path.join(output_dir, 'evaluation_summary.md')
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    print(summary)
    
    return report


def main():
    parser = argparse.ArgumentParser(description='Evaluate multi-task model')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--data_root', type=str, default='data', help='Root directory for datasets')
    parser.add_argument('--tasks', type=str, default='all', choices=['all', 'detection', 'segmentation', 'classification'],
                       help='Tasks to evaluate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args.weights, args.device)
    
    # Evaluate tasks
    if args.tasks == 'all':
        task_results = evaluate_all_tasks(model, args.data_root, args.batch_size, args.device)
    else:
        # Single task evaluation
        task_results = {}
        if args.tasks == 'detection':
            det_dataset = CocoDetectionDataset(
                root_dir=os.path.join(args.data_root, 'mini_coco_det'),
                split='val'
            )
            det_loader = torch.utils.data.DataLoader(
                det_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                collate_fn=CocoDetectionDataset.collate_fn
            )
            task_results['detection'] = evaluate_detection(model, det_loader, args.device)
        elif args.tasks == 'segmentation':
            seg_dataset = VOCSegmentationDataset(
                root_dir=os.path.join(args.data_root, 'mini_voc_seg'),
                split='val'
            )
            seg_loader = torch.utils.data.DataLoader(
                seg_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
            )
            task_results['segmentation'] = evaluate_segmentation(model, seg_loader, args.device)
        elif args.tasks == 'classification':
            cls_dataset = ImagenetteDataset(
                root_dir=os.path.join(args.data_root, 'imagenette_160'),
                split='val'
            )
            cls_loader = torch.utils.data.DataLoader(
                cls_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
            )
            task_results['classification'] = evaluate_classification(model, cls_loader, args.device)
    
    # Evaluate efficiency
    efficiency_results = evaluate_efficiency(model, args.device)
    
    # Compile results
    results = {
        'task_performance': task_results,
        'efficiency': efficiency_results,
    }
    
    # Check compliance
    results['compliance'] = check_compliance(results)
    
    # Generate report
    report = generate_report(results, args.output_dir)
    
    print(f"\nEvaluation complete! Results saved to {args.output_dir}")
    
    return results


if __name__ == '__main__':
    main()
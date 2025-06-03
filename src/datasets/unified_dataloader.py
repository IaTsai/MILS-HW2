import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import random
from enum import Enum


class TaskType(Enum):
    DETECTION = 'detection'
    SEGMENTATION = 'segmentation'
    CLASSIFICATION = 'classification'


def unified_collate_fn(batch):
    """Standalone collate function for UnifiedDataset with mixed task types"""
    images = []
    targets_list = []
    task_types = []
    
    for item in batch:
        images.append(item['images'])
        targets_list.append(item['targets'])
        task_types.append(item['task_type'])
    
    # Stack images
    images = torch.stack(images, 0)
    
    # Group by task type for easier processing
    task_groups = {}
    for i, task_type in enumerate(task_types):
        if task_type not in task_groups:
            task_groups[task_type] = []
        task_groups[task_type].append(i)
    
    return {
        'images': images,
        'task_types': task_types,
        'targets': targets_list,
        'task_groups': task_groups
    }


class UnifiedDataset(Dataset):
    """Unified dataset that combines detection, segmentation, and classification datasets
    
    Args:
        detection_dataset: Detection dataset (COCO format)
        segmentation_dataset: Segmentation dataset (VOC format)
        classification_dataset: Classification dataset (Imagenette)
        sampling_strategy: How to sample from different datasets
            - 'round_robin': Cycle through tasks in order
            - 'random': Random task selection
            - 'balanced': Equal probability for each task
            - 'weighted': Custom weights for each task
        task_weights: Weights for weighted sampling [det, seg, cls]
    """
    
    def __init__(self, detection_dataset=None, segmentation_dataset=None, 
                 classification_dataset=None, sampling_strategy='balanced',
                 task_weights=None):
        
        self.datasets = {}
        self.dataset_sizes = {}
        self.task_types = []
        
        if detection_dataset is not None:
            self._validate_dataset(detection_dataset, "detection_dataset")
            self.datasets[TaskType.DETECTION] = detection_dataset
            self.dataset_sizes[TaskType.DETECTION] = len(detection_dataset)
            self.task_types.append(TaskType.DETECTION)
        
        if segmentation_dataset is not None:
            self._validate_dataset(segmentation_dataset, "segmentation_dataset")
            self.datasets[TaskType.SEGMENTATION] = segmentation_dataset
            self.dataset_sizes[TaskType.SEGMENTATION] = len(segmentation_dataset)
            self.task_types.append(TaskType.SEGMENTATION)
        
        if classification_dataset is not None:
            self._validate_dataset(classification_dataset, "classification_dataset")
            self.datasets[TaskType.CLASSIFICATION] = classification_dataset
            self.dataset_sizes[TaskType.CLASSIFICATION] = len(classification_dataset)
            self.task_types.append(TaskType.CLASSIFICATION)
        
        if not self.datasets:
            raise ValueError("At least one dataset must be provided")
        
        self.sampling_strategy = sampling_strategy
        self.task_weights = task_weights or [1.0] * len(self.task_types)
        
        # Calculate total size based on strategy
        if sampling_strategy == 'round_robin':
            self.total_size = sum(self.dataset_sizes.values())
        else:
            # For other strategies, use the size of the largest dataset
            self.total_size = max(self.dataset_sizes.values()) * len(self.task_types)
        
        # Create index mapping for round-robin
        if sampling_strategy == 'round_robin':
            self.index_mapping = []
            for task, size in self.dataset_sizes.items():
                for i in range(size):
                    self.index_mapping.append((task, i))
            random.shuffle(self.index_mapping)
        
        print(f"UnifiedDataset created with {len(self.task_types)} tasks:")
        for task, size in self.dataset_sizes.items():
            print(f"  - {task.value}: {size} samples")
        print(f"Total size: {self.total_size}, Strategy: {sampling_strategy}")
    
    def _validate_dataset(self, dataset, param_name):
        """Validate that the dataset parameter is a proper Dataset object"""
        if isinstance(dataset, str):
            raise TypeError(f"{param_name} must be a Dataset object, not a string path. "
                          f"Please load the dataset first using the appropriate dataset class "
                          f"(e.g., CocoDetectionDataset, VOCSegmentationDataset, ImagenetteDataset).")
        
        # Check if it has __getitem__ and __len__ methods (basic Dataset interface)
        if not hasattr(dataset, '__getitem__'):
            raise TypeError(f"{param_name} must have a __getitem__ method (Dataset interface)")
        
        if not hasattr(dataset, '__len__'):
            raise TypeError(f"{param_name} must have a __len__ method (Dataset interface)")
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        # Determine task and actual index based on strategy
        if self.sampling_strategy == 'round_robin':
            task_type, actual_idx = self.index_mapping[idx % len(self.index_mapping)]
        
        elif self.sampling_strategy == 'random':
            task_type = random.choice(self.task_types)
            actual_idx = random.randint(0, self.dataset_sizes[task_type] - 1)
        
        elif self.sampling_strategy == 'balanced':
            # Cycle through tasks evenly
            task_idx = idx % len(self.task_types)
            task_type = self.task_types[task_idx]
            actual_idx = (idx // len(self.task_types)) % self.dataset_sizes[task_type]
        
        elif self.sampling_strategy == 'weighted':
            # Weighted random selection
            task_type = random.choices(self.task_types, weights=self.task_weights)[0]
            actual_idx = random.randint(0, self.dataset_sizes[task_type] - 1)
        
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
        
        # Get data from the selected dataset
        item = self.datasets[task_type][actual_idx]
        
        # Handle different return formats from individual datasets
        if isinstance(item, tuple) and len(item) == 2:
            img, target = item
        elif isinstance(item, dict):
            # Handle dict format (if any dataset returns this)
            img = item.get('image', item.get('img'))
            target = item.get('target', item.get('label'))
            if img is None or target is None:
                raise ValueError(f"Dataset {task_type} returned dict but missing 'image'/'img' or 'target'/'label' keys: {list(item.keys())}")
        else:
            raise ValueError(f"Dataset {task_type} returned unexpected format: {type(item)} (expected tuple or dict)")
        
        
        # Standardize target and add task info
        standardized_target = self._standardize_target(target, task_type)
        standardized_target['task_type'] = task_type.value
        
        # Return as unified dict format for direct DataLoader compatibility
        return {
            'images': img,
            'task_type': task_type.value,
            'targets': standardized_target
        }
    
    def _standardize_target(self, target, task_type):
        """Standardize target format across different tasks"""
        std_target = {}
        
        if task_type == TaskType.DETECTION:
            std_target['boxes'] = target.get('boxes', None)
            std_target['labels'] = target.get('labels', None)
            
        elif task_type == TaskType.SEGMENTATION:
            std_target['masks'] = target.get('masks', None)
            std_target['labels'] = target.get('labels', None)  # Unique classes in mask
            
        elif task_type == TaskType.CLASSIFICATION:
            std_target['labels'] = target.get('labels', None)
        
        # Add common fields
        std_target['image_id'] = target.get('image_id', torch.tensor([0]))
        
        return std_target


def create_unified_dataloaders(data_dir, batch_size=8, num_workers=4, 
                              train_split=0.6, val_split=0.2, test_split=0.2,
                              sampling_strategy='balanced', task_weights=None):
    """Factory function to create unified dataloaders
    
    Args:
        data_dir: Root directory containing datasets
        batch_size: Batch size for all loaders
        num_workers: Number of worker processes
        train_split: Fraction for training
        val_split: Fraction for validation  
        test_split: Fraction for testing
        sampling_strategy: Sampling strategy for unified loader
        task_weights: Weights for weighted sampling
        
    Returns:
        dict: Dictionary containing train/val/test unified dataloaders
    """
    from .coco_dataset import CocoDetectionDataset
    from .voc_dataset import VOCSegmentationDataset
    from .imagenette_dataset import ImagenetteDataset
    import os
    
    # Create datasets
    detection_train = CocoDetectionDataset(
        os.path.join(data_dir, "mini_coco_det"),
        split='train'
    )
    detection_val = CocoDetectionDataset(
        os.path.join(data_dir, "mini_coco_det"),
        split='val'
    )
    
    segmentation_train = VOCSegmentationDataset(
        os.path.join(data_dir, "mini_voc_seg"),
        split='train'
    )
    segmentation_val = VOCSegmentationDataset(
        os.path.join(data_dir, "mini_voc_seg"),
        split='val'
    )
    
    classification_train = ImagenetteDataset(
        os.path.join(data_dir, "imagenette_160"),
        split='train'
    )
    classification_val = ImagenetteDataset(
        os.path.join(data_dir, "imagenette_160"),
        split='val'
    )
    
    # Create unified dataloaders
    train_loader = UnifiedDataLoader(
        detection_dataset=detection_train,
        segmentation_dataset=segmentation_train,
        classification_dataset=classification_train,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        sampling_strategy=sampling_strategy,
        task_weights=task_weights
    )
    
    val_loader = UnifiedDataLoader(
        detection_dataset=detection_val,
        segmentation_dataset=segmentation_val,
        classification_dataset=classification_val,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        sampling_strategy=sampling_strategy,
        task_weights=task_weights
    )
    
    return {
        'train': train_loader,
        'val': val_loader
    }


class UnifiedDataLoader:
    """DataLoader wrapper for unified multi-task learning
    
    Provides both unified batches and task-specific batches
    """
    
    def __init__(self, detection_dataset=None, segmentation_dataset=None,
                 classification_dataset=None, batch_size=8, num_workers=4,
                 shuffle=True, sampling_strategy='balanced', task_weights=None,
                 pin_memory=True, drop_last=False):
        
        # Store individual datasets
        self.datasets = {
            TaskType.DETECTION: detection_dataset,
            TaskType.SEGMENTATION: segmentation_dataset,
            TaskType.CLASSIFICATION: classification_dataset
        }
        
        # Remove None datasets
        self.datasets = {k: v for k, v in self.datasets.items() if v is not None}
        
        # Create unified dataset
        self.unified_dataset = UnifiedDataset(
            detection_dataset, segmentation_dataset, classification_dataset,
            sampling_strategy=sampling_strategy, task_weights=task_weights
        )
        
        # Create unified loader
        self.unified_loader = DataLoader(
            self.unified_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=unified_collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last
        )
        
        # Create task-specific loaders
        self.task_loaders = {}
        for task_type, dataset in self.datasets.items():
            if dataset is not None:
                # Use dataset's custom collate_fn if available
                collate_fn = getattr(dataset, 'collate_fn', None)
                self.task_loaders[task_type] = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    collate_fn=collate_fn,
                    pin_memory=pin_memory,
                    drop_last=drop_last
                )
        
        # Create iterators for get_batch method
        self.task_iterators = {
            task: iter(loader) for task, loader in self.task_loaders.items()
        }
        
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def get_batch(self, task_type):
        """Get a batch for a specific task"""
        if isinstance(task_type, str):
            task_type = TaskType(task_type)
        
        try:
            batch = next(self.task_iterators[task_type])
        except StopIteration:
            # Restart iterator
            self.task_iterators[task_type] = iter(self.task_loaders[task_type])
            batch = next(self.task_iterators[task_type])
        
        return batch
    
    def __iter__(self):
        """Iterate over unified batches"""
        return iter(self.unified_loader)
    
    def __len__(self):
        """Length of unified loader"""
        return len(self.unified_loader)
    
    def get_task_loader(self, task_type):
        """Get loader for a specific task"""
        if isinstance(task_type, str):
            task_type = TaskType(task_type)
        return self.task_loaders.get(task_type, None)
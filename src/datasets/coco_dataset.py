import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import json
import os
from pycocotools.coco import COCO
from .transforms import get_detection_transforms


class CocoDetectionDataset(Dataset):
    """COCO format detection dataset
    
    Args:
        root_dir: Root directory of the dataset
        split: 'train' or 'val'
        transform: Optional transform to apply
        use_default_transform: Whether to use default transforms if transform is None
    """
    
    # Map original category IDs to continuous labels (0-9)
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light'
    ]
    
    def __init__(self, root_dir, split='train', transform=None, use_default_transform=True):
        self.root_dir = root_dir
        self.split = split
        
        ann_file = os.path.join(root_dir, 'annotations', f'{split}.json')
        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")
            
        self.coco = COCO(ann_file)
        self.img_ids = list(sorted(self.coco.imgs.keys()))
        
        # Get category IDs and create mapping
        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.label2cat = {i: cat_id for cat_id, i in self.cat2label.items()}
        
        # Set transform
        if transform is not None:
            self.transform = transform
        elif use_default_transform:
            self.transform = get_detection_transforms(train=(split == 'train'))
        else:
            self.transform = None
        
        print(f"Loaded COCO dataset: {len(self)} images, {len(self.cat_ids)} categories")
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, 'images', self.split, img_info['file_name'])
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Extract bboxes and labels
        bboxes = []
        labels = []
        areas = []
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            # Skip invalid boxes
            if w <= 0 or h <= 0:
                continue
                
            bboxes.append([x, y, w, h])  # Keep in xywh format, transform will convert
            labels.append(self.cat2label[ann['category_id']])
            areas.append(ann.get('area', w * h))
        
        # Convert to numpy arrays
        if len(bboxes) > 0:
            bboxes = np.array(bboxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
            areas = np.array(areas, dtype=np.float32)
        else:
            # No valid annotations, create empty arrays
            bboxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
            areas = np.zeros((0,), dtype=np.float32)
        
        # Create target dict
        target = {
            'boxes': torch.from_numpy(bboxes),
            'labels': torch.from_numpy(labels),
            'image_id': torch.tensor([img_id]),
            'area': torch.from_numpy(areas),
            'orig_size': torch.tensor([img.shape[0], img.shape[1]])
        }
        
        # Apply transforms
        if self.transform:
            img, target = self.transform(img, target)
        
        return img, target
    
    def get_img_info(self, idx):
        """Get image metadata"""
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        return img_info
    
    @staticmethod
    def collate_fn(batch):
        """Custom collate function for detection"""
        images = []
        targets = []
        
        for img, target in batch:
            images.append(img)
            targets.append(target)
        
        # Stack images into batch
        images = torch.stack(images, 0)
        
        return images, targets
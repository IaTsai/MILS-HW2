import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
from .transforms import get_segmentation_transforms


class VOCSegmentationDataset(Dataset):
    """Pascal VOC format segmentation dataset
    
    Args:
        root_dir: Root directory of the dataset
        split: 'train' or 'val'
        transform: Optional transform to apply
        use_default_transform: Whether to use default transforms if transform is None
    """
    
    # VOC class names and colors
    VOC_CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    # Color palette for visualization (RGB values)
    VOC_COLORMAP = [
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
        [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
        [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
        [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
        [0, 64, 128]
    ]
    
    def __init__(self, root_dir, split='train', transform=None, use_default_transform=True):
        self.root_dir = root_dir
        self.split = split
        
        self.image_dir = os.path.join(root_dir, 'JPEGImages')
        self.mask_dir = os.path.join(root_dir, 'SegmentationClass')
        
        split_file = os.path.join(root_dir, 'ImageSets', 'Segmentation', f'{split}.txt')
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
            
        with open(split_file, 'r') as f:
            self.image_names = [line.strip() for line in f.readlines() if line.strip()]
        
        self.num_classes = 21
        self.ignore_index = 255
        
        # Set transform
        if transform is not None:
            self.transform = transform
        elif use_default_transform:
            self.transform = get_segmentation_transforms(train=(split == 'train'))
        else:
            self.transform = None
            
        print(f"Loaded VOC dataset: {len(self)} images, {self.num_classes} classes")
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        
        img_path = os.path.join(self.image_dir, f'{img_name}.jpg')
        mask_path = os.path.join(self.mask_dir, f'{img_name}.png')
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        
        # Convert color mask to label indices
        mask_labels = self._decode_segmap(mask)
        
        # Create target dict
        target = {
            'masks': torch.from_numpy(mask_labels).long(),
            'labels': torch.unique(torch.from_numpy(mask_labels)),  # Unique classes in the mask
            'image_id': torch.tensor([idx]),
            'orig_size': torch.tensor([img.shape[0], img.shape[1]])
        }
        
        # Apply transforms
        if self.transform:
            img, target = self.transform(img, target)
        else:
            # Default conversion without augmentation
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        return img, target
    
    def _decode_segmap(self, mask_color):
        """Convert color mask to label indices"""
        mask_labels = np.zeros((mask_color.shape[0], mask_color.shape[1]), dtype=np.uint8)
        
        for label_idx, color in enumerate(self.VOC_COLORMAP):
            if label_idx >= self.num_classes:
                break
            # Match pixels with this color
            matches = np.all(mask_color == color, axis=2)
            mask_labels[matches] = label_idx
        
        # Set any unmatched pixels to ignore_index
        unmatched = np.sum(mask_labels == 0, axis=None) > 0
        if unmatched:
            # Check if there are non-background pixels that didn't match
            non_black = np.any(mask_color != [0, 0, 0], axis=2)
            unmatched_pixels = non_black & (mask_labels == 0)
            mask_labels[unmatched_pixels] = self.ignore_index
        
        return mask_labels
    
    def encode_segmap(self, mask_labels):
        """Convert label indices to color mask (for visualization)"""
        h, w = mask_labels.shape
        mask_color = np.zeros((h, w, 3), dtype=np.uint8)
        
        for label_idx in range(self.num_classes):
            mask_color[mask_labels == label_idx] = self.VOC_COLORMAP[label_idx]
        
        return mask_color
    
    def get_img_info(self, idx):
        """Get image metadata"""
        return {
            'file_name': f'{self.image_names[idx]}.jpg',
            'image_id': idx
        }
    
    @staticmethod
    def collate_fn(batch):
        """Custom collate function for segmentation"""
        images = []
        targets = []
        
        for img, target in batch:
            images.append(img)
            targets.append(target)
        
        # Stack images into batch
        images = torch.stack(images, 0)
        
        # Stack masks if they have the same size
        try:
            masks = torch.stack([t['masks'] for t in targets], 0)
            for i, target in enumerate(targets):
                target['masks'] = masks[i]
        except:
            # Keep masks as list if sizes differ
            pass
        
        return images, targets
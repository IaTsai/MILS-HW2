import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np
from .transforms import get_classification_transforms


class ImagenetteDataset(Dataset):
    """Imagenette classification dataset
    
    Args:
        root_dir: Root directory of the dataset
        split: 'train' or 'val'
        transform: Optional transform to apply
        use_default_transform: Whether to use default transforms if transform is None
    """
    
    # Imagenette class names (folder names to human-readable)
    IMAGENETTE_CLASSES = {
        'n01440764': 'tench',
        'n02102040': 'English springer',
        'n02979186': 'cassette player',
        'n03000684': 'chain saw',
        'n03028079': 'church',
        'n03394916': 'French horn',
        'n03417042': 'garbage truck',
        'n03425413': 'gas pump',
        'n03445777': 'golf ball',
        'n03888257': 'parachute'
    }
    
    def __init__(self, root_dir, split='train', transform=None, use_default_transform=True):
        self.root_dir = root_dir
        self.split = split
        
        self.data_dir = os.path.join(root_dir, split)
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Get class folders
        self.classes = sorted(os.listdir(self.data_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}
        
        # Get human-readable names
        self.class_names = [self.IMAGENETTE_CLASSES.get(cls, cls) for cls in self.classes]
        
        # Collect all image samples
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in sorted(os.listdir(class_dir)):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, self.class_to_idx[class_name]))
        
        # Set transform
        if transform is not None:
            self.transform = transform
        elif use_default_transform:
            self.transform = get_classification_transforms(train=(split == 'train'))
        else:
            self.transform = None
            
        print(f"Loaded Imagenette dataset: {len(self)} images, {len(self.classes)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create target dict for consistency with other datasets
        target = {
            'labels': torch.tensor(label, dtype=torch.long),
            'image_id': torch.tensor([idx]),
            'orig_size': torch.tensor([img.shape[0], img.shape[1]]),
            'class_name': self.class_names[label]
        }
        
        # Apply transforms
        if self.transform:
            # For classification, transform expects only image
            img = self.transform(img)
            # Standard transforms return only transformed image
            return img, target
        else:
            # Default conversion without augmentation
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            return img, target
    
    def get_img_info(self, idx):
        """Get image metadata"""
        img_path, label = self.samples[idx]
        return {
            'file_name': os.path.basename(img_path),
            'image_id': idx,
            'class_id': label,
            'class_name': self.class_names[label]
        }
    
    def get_class_names(self):
        """Get list of class names"""
        return self.class_names
    
    @staticmethod
    def collate_fn(batch):
        """Custom collate function for classification"""
        images = []
        targets = []
        
        for img, target in batch:
            images.append(img)
            targets.append(target)
        
        # Stack images into batch
        images = torch.stack(images, 0)
        
        # Stack labels
        labels = torch.stack([t['labels'] for t in targets], 0)
        
        # Create batch target
        batch_target = {
            'labels': labels,
            'image_ids': torch.cat([t['image_id'] for t in targets], 0),
            'class_names': [t['class_name'] for t in targets]
        }
        
        return images, batch_target
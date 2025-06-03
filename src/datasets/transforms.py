import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np
import random
import cv2


class Compose:
    """Compose multiple transforms together"""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    """Convert numpy image to tensor"""
    def __call__(self, image, target=None):
        image = F.to_tensor(image)
        return image, target


class Normalize:
    """Normalize image with mean and std"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class Resize:
    """Resize image and target"""
    def __init__(self, size):
        self.size = size
    
    def __call__(self, image, target=None):
        h, w = image.shape[-2:]
        image = F.resize(image, self.size)
        
        if target is not None:
            # Resize boxes
            if 'boxes' in target and target['boxes'] is not None:
                boxes = target['boxes']
                scale_x = self.size[1] / w
                scale_y = self.size[0] / h
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y
                target['boxes'] = boxes
            
            # Resize masks
            if 'masks' in target and target['masks'] is not None:
                masks = target['masks']
                masks = F.resize(masks.unsqueeze(0), self.size, interpolation=F.InterpolationMode.NEAREST)
                target['masks'] = masks.squeeze(0)
        
        return image, target


class RandomHorizontalFlip:
    """Random horizontal flip"""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, target=None):
        if random.random() < self.p:
            image = F.hflip(image)
            
            if target is not None:
                # Flip boxes
                if 'boxes' in target and target['boxes'] is not None:
                    boxes = target['boxes']
                    w = image.shape[-1]
                    boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                    target['boxes'] = boxes
                
                # Flip masks
                if 'masks' in target and target['masks'] is not None:
                    target['masks'] = F.hflip(target['masks'].unsqueeze(0)).squeeze(0)
        
        return image, target


class ColorJitter:
    """Random color jittering"""
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.color_jitter = T.ColorJitter(brightness, contrast, saturation, hue)
    
    def __call__(self, image, target=None):
        image = self.color_jitter(image)
        return image, target


class ConvertBoxFormat:
    """Convert bbox format from xywh to xyxy"""
    def __init__(self, source_format='xywh', target_format='xyxy'):
        self.source_format = source_format
        self.target_format = target_format
    
    def __call__(self, image, target=None):
        if target is not None and 'boxes' in target and target['boxes'] is not None:
            boxes = target['boxes']
            
            if self.source_format == 'xywh' and self.target_format == 'xyxy':
                # Convert [x, y, w, h] to [x1, y1, x2, y2]
                boxes_xyxy = boxes.clone()
                boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]
                boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]
                target['boxes'] = boxes_xyxy
            elif self.source_format == 'xyxy' and self.target_format == 'xywh':
                # Convert [x1, y1, x2, y2] to [x, y, w, h]
                boxes_xywh = boxes.clone()
                boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
                boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
                target['boxes'] = boxes_xywh
        
        return image, target


def get_detection_transforms(train=True, size=(512, 512)):
    """Get transforms for detection task"""
    if train:
        return Compose([
            ToTensor(),
            ConvertBoxFormat('xywh', 'xyxy'),
            Resize(size),
            RandomHorizontalFlip(0.5),
            ColorJitter(0.2, 0.2, 0.2, 0.1),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return Compose([
            ToTensor(),
            ConvertBoxFormat('xywh', 'xyxy'),
            Resize(size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def get_segmentation_transforms(train=True, size=(512, 512)):
    """Get transforms for segmentation task"""
    if train:
        return Compose([
            ToTensor(),
            Resize(size),
            RandomHorizontalFlip(0.5),
            ColorJitter(0.2, 0.2, 0.2, 0.1),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return Compose([
            ToTensor(),
            Resize(size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def get_classification_transforms(train=True, size=(512, 512)):
    """Get transforms for classification task"""
    if train:
        return T.Compose([
            T.ToPILImage(),
            T.Resize(size),
            T.RandomHorizontalFlip(0.5),
            T.ColorJitter(0.2, 0.2, 0.2, 0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.ToPILImage(),
            T.Resize(size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
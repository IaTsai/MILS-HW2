#!/usr/bin/env python3
import os
import zipfile
import tarfile
import json
import shutil
import argparse
from tqdm import tqdm
import requests
import time
import hashlib
import cv2
import numpy as np
from PIL import Image


def calculate_file_size(filepath):
    """Calculate file size in MB"""
    return os.path.getsize(filepath) / (1024 * 1024)


def download_file_with_retry(url, dest_path, max_retries=3):
    """Download file with retry mechanism"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(dest_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, 
                         desc=os.path.basename(dest_path)) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            return True
            
        except Exception as e:
            print(f"\nAttempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"Failed to download {url} after {max_retries} attempts")
                return False
    
    return False


def create_mock_coco_dataset(data_dir, train_count=240, val_count=60):
    """Create a mock Mini COCO dataset for testing"""
    print("\n=== Creating Mock Mini COCO Dataset ===")
    coco_dir = os.path.join(data_dir, 'mini_coco_det')
    
    # Create directory structure
    os.makedirs(os.path.join(coco_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(coco_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(coco_dir, 'annotations'), exist_ok=True)
    
    # Define COCO categories (subset)
    categories = [
        {"id": 1, "name": "person", "supercategory": "person"},
        {"id": 2, "name": "bicycle", "supercategory": "vehicle"},
        {"id": 3, "name": "car", "supercategory": "vehicle"},
        {"id": 4, "name": "motorcycle", "supercategory": "vehicle"},
        {"id": 5, "name": "airplane", "supercategory": "vehicle"},
        {"id": 6, "name": "bus", "supercategory": "vehicle"},
        {"id": 7, "name": "train", "supercategory": "vehicle"},
        {"id": 8, "name": "truck", "supercategory": "vehicle"},
        {"id": 9, "name": "boat", "supercategory": "vehicle"},
        {"id": 10, "name": "traffic light", "supercategory": "outdoor"}
    ]
    
    def create_split(split, count):
        images = []
        annotations = []
        ann_id = 1
        
        img_dir = os.path.join(coco_dir, 'images', split)
        
        with tqdm(total=count, desc=f"Creating {split} images") as pbar:
            for i in range(count):
                # Create random image
                img_id = i + 1
                width, height = np.random.choice([320, 416, 512]), np.random.choice([240, 320, 416])
                img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                
                # Add some simple shapes to make it more realistic
                num_objects = np.random.randint(1, 4)
                for _ in range(num_objects):
                    # Random rectangle
                    x1 = np.random.randint(0, width - 50)
                    y1 = np.random.randint(0, height - 50)
                    x2 = x1 + np.random.randint(30, min(100, width - x1))
                    y2 = y1 + np.random.randint(30, min(100, height - y1))
                    color = tuple(map(int, np.random.randint(0, 255, 3)))
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
                    
                    # Create annotation
                    annotations.append({
                        "id": int(ann_id),
                        "image_id": int(img_id),
                        "category_id": int(np.random.choice([cat['id'] for cat in categories])),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "area": float((x2 - x1) * (y2 - y1)),
                        "iscrowd": 0
                    })
                    ann_id += 1
                
                # Save image
                img_filename = f"{split}_{i:06d}.jpg"
                img_path = os.path.join(img_dir, img_filename)
                cv2.imwrite(img_path, img)
                
                # Add image info
                images.append({
                    "id": int(img_id),
                    "file_name": img_filename,
                    "width": int(width),
                    "height": int(height)
                })
                
                pbar.update(1)
        
        # Create COCO format annotation file
        coco_format = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
            "info": {
                "description": f"Mini COCO {split} dataset",
                "version": "1.0",
                "year": 2024
            }
        }
        
        ann_file = os.path.join(coco_dir, 'annotations', f'{split}.json')
        with open(ann_file, 'w') as f:
            json.dump(coco_format, f, indent=2)
        
        print(f"✓ Created {count} {split} images with {len(annotations)} annotations")
    
    # Create train and val splits
    create_split('train', train_count)
    create_split('val', val_count)
    
    # Calculate total size
    total_size = sum(calculate_file_size(os.path.join(root, file))
                    for root, _, files in os.walk(coco_dir)
                    for file in files)
    
    print(f"✓ Mini COCO dataset created: {total_size:.2f} MB")


def create_mock_voc_dataset(data_dir, train_count=240, val_count=60):
    """Create a mock Mini VOC dataset for testing"""
    print("\n=== Creating Mock Mini VOC Dataset ===")
    voc_dir = os.path.join(data_dir, 'mini_voc_seg')
    
    # Create directory structure
    os.makedirs(os.path.join(voc_dir, 'JPEGImages'), exist_ok=True)
    os.makedirs(os.path.join(voc_dir, 'SegmentationClass'), exist_ok=True)
    os.makedirs(os.path.join(voc_dir, 'ImageSets', 'Segmentation'), exist_ok=True)
    
    # VOC classes (subset)
    voc_classes = {
        0: (0, 0, 0),        # background
        1: (128, 0, 0),      # aeroplane
        2: (0, 128, 0),      # bicycle
        3: (128, 128, 0),    # bird
        4: (0, 0, 128),      # boat
        5: (128, 0, 128),    # bottle
        6: (0, 128, 128),    # bus
        7: (128, 128, 128),  # car
        8: (64, 0, 0),       # cat
        9: (192, 0, 0),      # chair
        10: (64, 128, 0),    # cow
        255: (224, 224, 192) # void/unlabeled
    }
    
    all_image_names = []
    
    def create_images(split, count):
        image_names = []
        
        with tqdm(total=count, desc=f"Creating {split} images") as pbar:
            for i in range(count):
                # Image name
                img_name = f"{split}_{i:06d}"
                image_names.append(img_name)
                
                # Create random image
                width, height = 320, 240
                img = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
                
                # Create segmentation mask
                mask = np.zeros((height, width), dtype=np.uint8)
                
                # Add random segments
                num_segments = np.random.randint(2, 5)
                for _ in range(num_segments):
                    # Random polygon or rectangle
                    class_id = int(np.random.choice(list(voc_classes.keys())[1:-1]))  # Exclude bg and void
                    
                    if np.random.random() > 0.5:
                        # Rectangle
                        x1 = np.random.randint(0, width - 50)
                        y1 = np.random.randint(0, height - 50)
                        x2 = x1 + np.random.randint(30, min(100, width - x1))
                        y2 = y1 + np.random.randint(30, min(100, height - y1))
                        mask[y1:y2, x1:x2] = class_id
                        cv2.rectangle(img, (x1, y1), (x2, y2), voc_classes[class_id], -1)
                    else:
                        # Circle
                        cx = np.random.randint(30, width - 30)
                        cy = np.random.randint(30, height - 30)
                        radius = np.random.randint(15, 40)
                        cv2.circle(mask, (cx, cy), radius, class_id, -1)
                        cv2.circle(img, (cx, cy), radius, voc_classes[class_id], -1)
                
                # Save image
                img_path = os.path.join(voc_dir, 'JPEGImages', f'{img_name}.jpg')
                cv2.imwrite(img_path, img)
                
                # Save mask as color image
                mask_color = np.zeros((height, width, 3), dtype=np.uint8)
                for class_id, color in voc_classes.items():
                    mask_color[mask == class_id] = color
                
                mask_path = os.path.join(voc_dir, 'SegmentationClass', f'{img_name}.png')
                cv2.imwrite(mask_path, mask_color)
                
                pbar.update(1)
        
        # Write split file
        split_file = os.path.join(voc_dir, 'ImageSets', 'Segmentation', f'{split}.txt')
        with open(split_file, 'w') as f:
            f.write('\n'.join(image_names))
        
        return image_names
    
    # Create train and val splits
    train_names = create_images('train', train_count)
    val_names = create_images('val', val_count)
    all_image_names = train_names + val_names
    
    # Write trainval file
    trainval_file = os.path.join(voc_dir, 'ImageSets', 'Segmentation', 'trainval.txt')
    with open(trainval_file, 'w') as f:
        f.write('\n'.join(all_image_names))
    
    # Calculate total size
    total_size = sum(calculate_file_size(os.path.join(root, file))
                    for root, _, files in os.walk(voc_dir)
                    for file in files)
    
    print(f"✓ Mini VOC dataset created: {total_size:.2f} MB")


def create_mock_imagenette_dataset(data_dir, train_count=240, val_count=60):
    """Create a mock Imagenette-160 dataset for testing"""
    print("\n=== Creating Mock Imagenette-160 Dataset ===")
    imagenette_dir = os.path.join(data_dir, 'imagenette_160')
    
    # Imagenette class names
    class_names = [
        'n01440764',  # tench
        'n02102040',  # English springer
        'n02979186',  # cassette player
        'n03000684',  # chain saw
        'n03028079',  # church
        'n03394916',  # French horn
        'n03417042',  # garbage truck
        'n03425413',  # gas pump
        'n03445777',  # golf ball
        'n03888257',  # parachute
    ]
    
    def create_split(split, count):
        split_dir = os.path.join(imagenette_dir, split)
        images_per_class = count // len(class_names)
        extra = count % len(class_names)
        
        with tqdm(total=count, desc=f"Creating {split} images") as pbar:
            for i, class_name in enumerate(class_names):
                class_dir = os.path.join(split_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                
                # Determine number of images for this class
                n_images = images_per_class + (1 if i < extra else 0)
                
                for j in range(n_images):
                    # Create random image (160x160 as per dataset name)
                    img = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
                    
                    # Add some texture/pattern
                    if np.random.random() > 0.5:
                        # Add gradient
                        gradient = np.linspace(0, 255, 160)
                        img[:, :, 0] = gradient.reshape(1, -1)
                    else:
                        # Add noise pattern
                        noise = np.random.normal(128, 30, (160, 160))
                        img[:, :, 1] = np.clip(noise, 0, 255)
                    
                    # Save image
                    img_filename = f"{class_name}_{j:04d}.JPEG"
                    img_path = os.path.join(class_dir, img_filename)
                    Image.fromarray(img).save(img_path, 'JPEG', quality=85)
                    
                    pbar.update(1)
    
    # Create train and val splits
    create_split('train', train_count)
    create_split('val', val_count)
    
    # Calculate total size
    total_size = sum(calculate_file_size(os.path.join(root, file))
                    for root, _, files in os.walk(imagenette_dir)
                    for file in files)
    
    print(f"✓ Imagenette-160 dataset created: {total_size:.2f} MB")


def download_mini_coco(data_dir):
    """Download or create Mini COCO dataset"""
    print("\n" + "="*50)
    print("Mini COCO Dataset")
    print("="*50)
    
    coco_dir = os.path.join(data_dir, 'mini_coco_det')
    
    # Check if already exists
    if os.path.exists(coco_dir) and len(os.listdir(coco_dir)) > 0:
        print("✓ Mini COCO dataset already exists")
        return
    
    # Try to download from URL (placeholder)
    urls = {
        'dataset': 'https://example.com/mini_coco.zip'  # Replace with actual URL
    }
    
    print("Note: Using mock data generation for Mini COCO dataset")
    print("Replace with actual download URL when available")
    
    # Create mock dataset
    create_mock_coco_dataset(data_dir)


def download_mini_voc(data_dir):
    """Download or create Mini VOC dataset"""
    print("\n" + "="*50)
    print("Mini VOC Segmentation Dataset")
    print("="*50)
    
    voc_dir = os.path.join(data_dir, 'mini_voc_seg')
    
    # Check if already exists
    if os.path.exists(voc_dir) and len(os.listdir(voc_dir)) > 0:
        print("✓ Mini VOC dataset already exists")
        return
    
    # Try to download from URL (placeholder)
    urls = {
        'dataset': 'https://example.com/mini_voc.zip'  # Replace with actual URL
    }
    
    print("Note: Using mock data generation for Mini VOC dataset")
    print("Replace with actual download URL when available")
    
    # Create mock dataset
    create_mock_voc_dataset(data_dir)


def download_imagenette(data_dir):
    """Download or create Imagenette-160 dataset"""
    print("\n" + "="*50)
    print("Imagenette-160 Dataset")
    print("="*50)
    
    imagenette_dir = os.path.join(data_dir, 'imagenette_160')
    
    # Check if already exists
    if os.path.exists(imagenette_dir) and len(os.listdir(imagenette_dir)) > 0:
        print("✓ Imagenette-160 dataset already exists")
        return
    
    # For Imagenette, we'll create a smaller subset
    print("Note: Creating a subset of Imagenette-160 for testing")
    print("For full dataset, use: https://github.com/fastai/imagenette")
    
    # Create mock dataset
    create_mock_imagenette_dataset(data_dir)


def check_total_size(data_dir):
    """Check total size of all datasets"""
    total_size = 0
    dataset_sizes = {}
    
    for dataset_name in ['mini_coco_det', 'mini_voc_seg', 'imagenette_160']:
        dataset_path = os.path.join(data_dir, dataset_name)
        if os.path.exists(dataset_path):
            size = sum(calculate_file_size(os.path.join(root, file))
                      for root, _, files in os.walk(dataset_path)
                      for file in files)
            dataset_sizes[dataset_name] = size
            total_size += size
    
    print("\n" + "="*50)
    print("Dataset Size Summary")
    print("="*50)
    
    for dataset, size in dataset_sizes.items():
        print(f"{dataset}: {size:.2f} MB")
    
    print(f"\nTotal size: {total_size:.2f} MB")
    
    if total_size > 120:
        print(f"⚠️  WARNING: Total size exceeds 120 MB limit!")
    else:
        print(f"✓ Total size is within 120 MB limit")
    
    return total_size <= 120


def main():
    parser = argparse.ArgumentParser(description='Download datasets for multi-task learning')
    parser.add_argument('--data-dir', type=str, default='./data', 
                        help='Directory to save datasets')
    parser.add_argument('--dataset', type=str, choices=['all', 'coco', 'voc', 'imagenette'],
                        default='all', help='Which dataset to download')
    parser.add_argument('--force', action='store_true',
                        help='Force re-download even if datasets exist')
    
    args = parser.parse_args()
    
    # Convert to absolute path
    args.data_dir = os.path.abspath(args.data_dir)
    
    print(f"Data directory: {args.data_dir}")
    os.makedirs(args.data_dir, exist_ok=True)
    
    try:
        if args.dataset in ['all', 'coco']:
            download_mini_coco(args.data_dir)
        
        if args.dataset in ['all', 'voc']:
            download_mini_voc(args.data_dir)
        
        if args.dataset in ['all', 'imagenette']:
            download_imagenette(args.data_dir)
        
        # Check total size
        if args.dataset == 'all':
            size_ok = check_total_size(args.data_dir)
            if not size_ok:
                print("\n❌ Dataset size exceeds limit!")
                return 1
        
        print("\n✅ All datasets downloaded/created successfully!")
        print(f"\nNext step: Run verification script")
        print(f"python scripts/verify_data.py --data-dir {args.data_dir}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
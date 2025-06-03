#!/usr/bin/env python3
import os
import json
import cv2
import argparse
from tqdm import tqdm
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_dataset_size(dataset_path):
    """Calculate total size of dataset in MB"""
    total_size = 0
    for root, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(root, file)
            total_size += os.path.getsize(file_path)
    return total_size / (1024 * 1024)  # Convert to MB


def verify_coco_dataset(coco_dir):
    """Comprehensive verification of COCO dataset"""
    print("\n" + "="*50)
    print("Verifying Mini COCO Dataset")
    print("="*50)
    
    stats = {
        'valid': True,
        'train_images': 0,
        'val_images': 0,
        'train_annotations': 0,
        'val_annotations': 0,
        'categories': set(),
        'image_sizes': [],
        'bbox_sizes': [],
        'size_mb': 0
    }
    
    # Check directory structure
    required_dirs = ['images/train', 'images/val', 'annotations']
    for dir_name in required_dirs:
        dir_path = os.path.join(coco_dir, dir_name)
        if not os.path.exists(dir_path):
            print(f"❌ Missing directory: {dir_path}")
            stats['valid'] = False
        else:
            print(f"✓ Found directory: {dir_path}")
    
    # Check and analyze annotations
    ann_files = ['train.json', 'val.json']
    for ann_file in ann_files:
        ann_path = os.path.join(coco_dir, 'annotations', ann_file)
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                data = json.load(f)
                
                split = ann_file.replace('.json', '')
                num_images = len(data['images'])
                num_annotations = len(data['annotations'])
                
                if split == 'train':
                    stats['train_images'] = num_images
                    stats['train_annotations'] = num_annotations
                else:
                    stats['val_images'] = num_images
                    stats['val_annotations'] = num_annotations
                
                # Collect category statistics
                for ann in data['annotations']:
                    stats['categories'].add(ann['category_id'])
                    stats['bbox_sizes'].append(ann['area'])
                
                # Collect image size statistics
                for img in data['images']:
                    stats['image_sizes'].append((img['width'], img['height']))
                
                print(f"✓ {ann_file}: {num_images} images, {num_annotations} annotations")
                
                # Verify expected counts
                expected = 240 if split == 'train' else 60
                if num_images != expected:
                    print(f"  ⚠️  Warning: Expected {expected} images, found {num_images}")
        else:
            print(f"❌ Missing annotation file: {ann_path}")
            stats['valid'] = False
    
    # Verify actual image files
    for split in ['train', 'val']:
        img_dir = os.path.join(coco_dir, 'images', split)
        if os.path.exists(img_dir):
            actual_images = len([f for f in os.listdir(img_dir) 
                               if f.endswith(('.jpg', '.jpeg', '.png'))])
            expected = 240 if split == 'train' else 60
            print(f"✓ {split} images directory: {actual_images} files")
            if actual_images != expected:
                print(f"  ⚠️  Warning: Expected {expected} images, found {actual_images}")
    
    # Calculate dataset size
    stats['size_mb'] = calculate_dataset_size(coco_dir)
    print(f"\nDataset size: {stats['size_mb']:.2f} MB")
    
    # Print statistics
    if stats['valid']:
        print(f"\nStatistics:")
        print(f"  - Categories: {len(stats['categories'])} unique")
        print(f"  - Avg annotations per image: {(stats['train_annotations'] + stats['val_annotations']) / (stats['train_images'] + stats['val_images']):.2f}")
        
        if stats['image_sizes']:
            widths = [s[0] for s in stats['image_sizes']]
            heights = [s[1] for s in stats['image_sizes']]
            print(f"  - Image sizes: {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}")
        
        if stats['bbox_sizes']:
            print(f"  - BBox areas: min={min(stats['bbox_sizes']):.0f}, max={max(stats['bbox_sizes']):.0f}, avg={np.mean(stats['bbox_sizes']):.0f}")
    
    return stats['valid'], stats


def verify_voc_dataset(voc_dir):
    """Comprehensive verification of VOC dataset"""
    print("\n" + "="*50)
    print("Verifying Mini VOC Segmentation Dataset")
    print("="*50)
    
    stats = {
        'valid': True,
        'train_images': 0,
        'val_images': 0,
        'classes': set(),
        'image_sizes': [],
        'size_mb': 0
    }
    
    # Check directory structure
    required_dirs = ['JPEGImages', 'SegmentationClass', 'ImageSets/Segmentation']
    for dir_name in required_dirs:
        dir_path = os.path.join(voc_dir, dir_name)
        if not os.path.exists(dir_path):
            print(f"❌ Missing directory: {dir_path}")
            stats['valid'] = False
        else:
            print(f"✓ Found directory: {dir_path}")
    
    # Check split files and count images
    split_files = ['train.txt', 'val.txt']
    for split_file in split_files:
        split_path = os.path.join(voc_dir, 'ImageSets/Segmentation', split_file)
        if os.path.exists(split_path):
            with open(split_path, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                split = split_file.replace('.txt', '')
                
                if split == 'train':
                    stats['train_images'] = len(lines)
                else:
                    stats['val_images'] = len(lines)
                
                print(f"✓ {split_file}: {len(lines)} images")
                
                # Verify expected counts
                expected = 240 if split == 'train' else 60
                if len(lines) != expected:
                    print(f"  ⚠️  Warning: Expected {expected} images, found {len(lines)}")
                
                # Sample check: verify files exist
                for img_name in lines[:5]:  # Check first 5
                    img_path = os.path.join(voc_dir, 'JPEGImages', f'{img_name}.jpg')
                    mask_path = os.path.join(voc_dir, 'SegmentationClass', f'{img_name}.png')
                    
                    if not os.path.exists(img_path):
                        print(f"  ❌ Missing image: {img_path}")
                        stats['valid'] = False
                    if not os.path.exists(mask_path):
                        print(f"  ❌ Missing mask: {mask_path}")
                        stats['valid'] = False
                    else:
                        # Analyze mask for class statistics
                        mask = cv2.imread(mask_path)
                        if mask is not None:
                            unique_colors = np.unique(mask.reshape(-1, mask.shape[2]), axis=0)
                            for color in unique_colors:
                                stats['classes'].add(tuple(color))
                            stats['image_sizes'].append(mask.shape[:2])
        else:
            print(f"❌ Missing split file: {split_path}")
            stats['valid'] = False
    
    # Calculate dataset size
    stats['size_mb'] = calculate_dataset_size(voc_dir)
    print(f"\nDataset size: {stats['size_mb']:.2f} MB")
    
    # Print statistics
    if stats['valid']:
        print(f"\nStatistics:")
        print(f"  - Total images: {stats['train_images'] + stats['val_images']}")
        print(f"  - Unique segmentation classes: {len(stats['classes'])}")
        
        if stats['image_sizes']:
            heights = [s[0] for s in stats['image_sizes']]
            widths = [s[1] for s in stats['image_sizes']]
            print(f"  - Image sizes: {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}")
    
    return stats['valid'], stats


def verify_imagenette_dataset(imagenette_dir):
    """Comprehensive verification of Imagenette dataset"""
    print("\n" + "="*50)
    print("Verifying Imagenette-160 Dataset")
    print("="*50)
    
    stats = {
        'valid': True,
        'train_images': 0,
        'val_images': 0,
        'classes': {},
        'image_sizes': [],
        'size_mb': 0
    }
    
    # Expected class names
    expected_classes = [
        'n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079',
        'n03394916', 'n03417042', 'n03425413', 'n03445777', 'n03888257'
    ]
    
    # Check directory structure
    required_dirs = ['train', 'val']
    for dir_name in required_dirs:
        dir_path = os.path.join(imagenette_dir, dir_name)
        if not os.path.exists(dir_path):
            print(f"❌ Missing directory: {dir_path}")
            stats['valid'] = False
        else:
            print(f"✓ Found directory: {dir_path}")
            
            # Count images per class
            classes = os.listdir(dir_path)
            total_images = 0
            class_counts = {}
            
            for class_name in classes:
                class_dir = os.path.join(dir_path, class_name)
                if os.path.isdir(class_dir):
                    images = [f for f in os.listdir(class_dir) 
                             if f.endswith(('.jpg', '.jpeg', '.png', '.JPEG'))]
                    num_images = len(images)
                    total_images += num_images
                    class_counts[class_name] = num_images
                    
                    # Sample image size check
                    if images and len(stats['image_sizes']) < 10:
                        img_path = os.path.join(class_dir, images[0])
                        img = cv2.imread(img_path)
                        if img is not None:
                            stats['image_sizes'].append(img.shape[:2])
            
            if dir_name == 'train':
                stats['train_images'] = total_images
            else:
                stats['val_images'] = total_images
            
            stats['classes'][dir_name] = class_counts
            
            print(f"  - {len(classes)} classes, {total_images} total images")
            
            # Verify expected counts
            expected = 240 if dir_name == 'train' else 60
            if total_images != expected:
                print(f"  ⚠️  Warning: Expected {expected} images, found {total_images}")
            
            # Check class balance
            if class_counts:
                min_per_class = min(class_counts.values())
                max_per_class = max(class_counts.values())
                print(f"  - Images per class: {min_per_class} to {max_per_class}")
            
            # Verify expected classes
            missing_classes = set(expected_classes) - set(classes)
            if missing_classes:
                print(f"  ⚠️  Missing expected classes: {missing_classes}")
    
    # Calculate dataset size
    stats['size_mb'] = calculate_dataset_size(imagenette_dir)
    print(f"\nDataset size: {stats['size_mb']:.2f} MB")
    
    # Print statistics
    if stats['valid']:
        print(f"\nStatistics:")
        print(f"  - Total images: {stats['train_images'] + stats['val_images']}")
        
        if stats['image_sizes']:
            heights = [s[0] for s in stats['image_sizes']]
            widths = [s[1] for s in stats['image_sizes']]
            print(f"  - Image sizes: {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}")
            avg_size = (np.mean(widths), np.mean(heights))
            print(f"  - Average size: {avg_size[0]:.0f}x{avg_size[1]:.0f}")
    
    return stats['valid'], stats


def generate_dataset_report(data_dir, all_stats):
    """Generate a comprehensive dataset report"""
    print("\n" + "="*70)
    print("DATASET VERIFICATION REPORT")
    print("="*70)
    
    total_size = sum(stats.get('size_mb', 0) for _, stats in all_stats.values())
    total_images = 0
    
    # Summary table
    print("\nDataset Summary:")
    print("-" * 70)
    print(f"{'Dataset':<20} {'Status':<10} {'Train':<10} {'Val':<10} {'Size (MB)':<10}")
    print("-" * 70)
    
    for dataset_name, (valid, stats) in all_stats.items():
        status = "✓ PASS" if valid else "✗ FAIL"
        train = stats.get('train_images', 0)
        val = stats.get('val_images', 0)
        size = stats.get('size_mb', 0)
        total_images += train + val
        
        print(f"{dataset_name:<20} {status:<10} {train:<10} {val:<10} {size:<10.2f}")
    
    print("-" * 70)
    print(f"{'TOTAL':<20} {'':<10} {'':<10} {'':<10} {total_size:<10.2f}")
    print("-" * 70)
    
    # Size check
    print(f"\nTotal dataset size: {total_size:.2f} MB")
    if total_size <= 120:
        print("✅ Size is within 120 MB limit")
    else:
        print(f"❌ Size exceeds 120 MB limit by {total_size - 120:.2f} MB")
    
    # Image count check
    expected_total = 300 * 3  # 300 images per dataset
    print(f"\nTotal images: {total_images} (expected: {expected_total})")
    if total_images == expected_total:
        print("✅ Image count matches expected")
    else:
        print(f"⚠️  Image count mismatch: {total_images - expected_total:+d}")
    
    # Save report to file
    report_path = os.path.join(data_dir, 'dataset_verification_report.txt')
    with open(report_path, 'w') as f:
        f.write("DATASET VERIFICATION REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated at: {os.path.abspath(data_dir)}\n")
        f.write(f"Total size: {total_size:.2f} MB\n")
        f.write(f"Total images: {total_images}\n")
        f.write("\nDataset Details:\n")
        for dataset_name, (valid, stats) in all_stats.items():
            f.write(f"\n{dataset_name}:\n")
            f.write(f"  Valid: {valid}\n")
            f.write(f"  Train images: {stats.get('train_images', 0)}\n")
            f.write(f"  Val images: {stats.get('val_images', 0)}\n")
            f.write(f"  Size: {stats.get('size_mb', 0):.2f} MB\n")
    
    print(f"\n📄 Detailed report saved to: {report_path}")
    
    return total_size <= 120 and all(valid for valid, _ in all_stats.values())


def verify_sample_images(data_dir):
    """Verify that sample images can be loaded and displayed"""
    print("\n" + "="*50)
    print("Sample Image Verification")
    print("="*50)
    
    samples = []
    
    # Test COCO image
    coco_train_dir = os.path.join(data_dir, 'mini_coco_det/images/train')
    if os.path.exists(coco_train_dir):
        images = [f for f in os.listdir(coco_train_dir) if f.endswith(('.jpg', '.jpeg', '.png'))][:1]
        if images:
            img_path = os.path.join(coco_train_dir, images[0])
            img = cv2.imread(img_path)
            if img is not None:
                print(f"✓ COCO sample image loaded: {img.shape}")
                samples.append(('COCO Detection', img))
            else:
                print(f"❌ Failed to load COCO image: {img_path}")
    
    # Test VOC image and mask
    voc_img_dir = os.path.join(data_dir, 'mini_voc_seg/JPEGImages')
    voc_mask_dir = os.path.join(data_dir, 'mini_voc_seg/SegmentationClass')
    if os.path.exists(voc_img_dir):
        images = [f for f in os.listdir(voc_img_dir) if f.endswith('.jpg')][:1]
        if images:
            img_name = images[0].replace('.jpg', '')
            img_path = os.path.join(voc_img_dir, images[0])
            mask_path = os.path.join(voc_mask_dir, f'{img_name}.png')
            
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path)
            
            if img is not None:
                print(f"✓ VOC sample image loaded: {img.shape}")
                samples.append(('VOC Image', img))
            else:
                print(f"❌ Failed to load VOC image: {img_path}")
            
            if mask is not None:
                print(f"✓ VOC sample mask loaded: {mask.shape}")
                samples.append(('VOC Mask', mask))
            else:
                print(f"❌ Failed to load VOC mask: {mask_path}")
    
    # Test Imagenette image
    imagenette_train_dir = os.path.join(data_dir, 'imagenette_160/train')
    if os.path.exists(imagenette_train_dir):
        classes = os.listdir(imagenette_train_dir)
        if classes:
            class_dir = os.path.join(imagenette_train_dir, classes[0])
            images = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.JPEG'))][:1]
            if images:
                img_path = os.path.join(class_dir, images[0])
                img = cv2.imread(img_path)
                if img is not None:
                    print(f"✓ Imagenette sample image loaded: {img.shape}")
                    samples.append(('Imagenette', img))
                else:
                    print(f"❌ Failed to load Imagenette image: {img_path}")
    
    # Create visualization if samples loaded
    if samples and len(samples) >= 3:
        try:
            fig, axes = plt.subplots(1, len(samples), figsize=(15, 5))
            if len(samples) == 1:
                axes = [axes]
            
            for ax, (title, img) in zip(axes, samples):
                if len(img.shape) == 3:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img_rgb = img
                ax.imshow(img_rgb)
                ax.set_title(title)
                ax.axis('off')
            
            plt.tight_layout()
            sample_path = os.path.join(data_dir, 'sample_images.png')
            plt.savefig(sample_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"\n📷 Sample images saved to: {sample_path}")
        except Exception as e:
            print(f"Warning: Could not save sample images: {e}")


def main():
    parser = argparse.ArgumentParser(description='Verify datasets for multi-task learning')
    parser.add_argument('--data-dir', type=str, default='./data', 
                        help='Directory containing datasets')
    parser.add_argument('--detailed', action='store_true',
                        help='Show detailed statistics')
    
    args = parser.parse_args()
    
    # Convert to absolute path
    args.data_dir = os.path.abspath(args.data_dir)
    
    print(f"Verifying datasets in: {args.data_dir}")
    
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory does not exist: {args.data_dir}")
        print(f"Please run: python scripts/download_data.py --data-dir {args.data_dir}")
        return 1
    
    all_stats = {}
    
    # Verify each dataset
    coco_dir = os.path.join(args.data_dir, 'mini_coco_det')
    if os.path.exists(coco_dir):
        valid, stats = verify_coco_dataset(coco_dir)
        all_stats['mini_coco_det'] = (valid, stats)
    else:
        print(f"\n❌ Mini COCO dataset not found at: {coco_dir}")
        all_stats['mini_coco_det'] = (False, {'size_mb': 0})
    
    voc_dir = os.path.join(args.data_dir, 'mini_voc_seg')
    if os.path.exists(voc_dir):
        valid, stats = verify_voc_dataset(voc_dir)
        all_stats['mini_voc_seg'] = (valid, stats)
    else:
        print(f"\n❌ Mini VOC dataset not found at: {voc_dir}")
        all_stats['mini_voc_seg'] = (False, {'size_mb': 0})
    
    imagenette_dir = os.path.join(args.data_dir, 'imagenette_160')
    if os.path.exists(imagenette_dir):
        valid, stats = verify_imagenette_dataset(imagenette_dir)
        all_stats['imagenette_160'] = (valid, stats)
    else:
        print(f"\n❌ Imagenette dataset not found at: {imagenette_dir}")
        all_stats['imagenette_160'] = (False, {'size_mb': 0})
    
    # Verify sample images
    verify_sample_images(args.data_dir)
    
    # Generate report
    all_valid = generate_dataset_report(args.data_dir, all_stats)
    
    if all_valid:
        print("\n✅ All datasets verified successfully!")
        print("\nNext step: Run Phase 1 check")
        print("python scripts/phase1_check.py")
        return 0
    else:
        print("\n❌ Dataset verification failed!")
        print("Please check the errors above and re-run download script if needed.")
        return 1


if __name__ == '__main__':
    exit(main())
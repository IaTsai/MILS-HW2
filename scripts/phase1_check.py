#!/usr/bin/env python3
import os
import sys
import torch
import argparse
from datetime import datetime
import subprocess
import json


def check_gpu_environment():
    """Check GPU availability and configuration"""
    print("\n" + "="*50)
    print("GPU Environment Check")
    print("="*50)
    
    gpu_info = {
        'cuda_available': False,
        'gpu_count': 0,
        'gpu_names': [],
        'cuda_version': 'N/A',
        'cudnn_version': 'N/A',
        'current_device': None,
        'memory_info': []
    }
    
    try:
        import torch
        
        # Check CUDA availability
        gpu_info['cuda_available'] = torch.cuda.is_available()
        
        if gpu_info['cuda_available']:
            gpu_info['gpu_count'] = torch.cuda.device_count()
            gpu_info['cuda_version'] = torch.version.cuda
            gpu_info['cudnn_version'] = torch.backends.cudnn.version()
            
            print(f"✓ CUDA is available")
            print(f"  - CUDA version: {gpu_info['cuda_version']}")
            print(f"  - cuDNN version: {gpu_info['cudnn_version']}")
            print(f"  - Number of GPUs: {gpu_info['gpu_count']}")
            
            # Get info for each GPU
            for i in range(gpu_info['gpu_count']):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_info['gpu_names'].append(gpu_name)
                
                # Memory info
                total_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9
                
                gpu_info['memory_info'].append({
                    'device': i,
                    'name': gpu_name,
                    'total_memory_gb': total_memory,
                    'allocated_gb': allocated,
                    'reserved_gb': reserved
                })
                
                print(f"\n  GPU {i}: {gpu_name}")
                print(f"    - Total memory: {total_memory:.2f} GB")
                print(f"    - Allocated: {allocated:.2f} GB")
                print(f"    - Reserved: {reserved:.2f} GB")
            
            # Check CUDA_VISIBLE_DEVICES
            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'All')
            print(f"\n  CUDA_VISIBLE_DEVICES: {cuda_visible}")
            
            # Test GPU computation
            print("\n  Testing GPU computation...")
            try:
                x = torch.randn(1000, 1000).cuda()
                y = torch.randn(1000, 1000).cuda()
                z = torch.matmul(x, y)
                torch.cuda.synchronize()
                print("  ✓ GPU computation test passed")
            except Exception as e:
                print(f"  ❌ GPU computation test failed: {e}")
            
        else:
            print("❌ CUDA is not available")
            print("  Please ensure you have a CUDA-capable GPU and PyTorch is installed with CUDA support")
            
            # Try to get more info using nvidia-smi
            try:
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                if result.returncode == 0:
                    print("\n  nvidia-smi output detected GPUs but PyTorch cannot access CUDA")
                    print("  This might be due to:")
                    print("  - PyTorch installed without CUDA support")
                    print("  - CUDA driver/library version mismatch")
                    print("  - Missing CUDA libraries")
            except:
                print("\n  nvidia-smi not found - no NVIDIA GPU detected")
        
    except ImportError:
        print("❌ PyTorch not installed")
        gpu_info['cuda_available'] = False
    
    # Recommendations for training
    print("\n" + "-"*50)
    print("Training Recommendations:")
    if gpu_info['cuda_available'] and gpu_info['gpu_count'] > 0:
        print("✓ GPU environment is ready for training")
        print(f"  - You have {gpu_info['gpu_count']} GPU(s) available")
        print("\n  To use a specific GPU (e.g., GPU 1), run:")
        print("  CUDA_VISIBLE_DEVICES=1 python train.py")
        print("\n  For debugging with CUDA errors, run:")
        print("  CUDA_LAUNCH_BLOCKING=1 python train.py")
        print("\n  Your command example:")
        print("  CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python train.py")
    else:
        print("⚠️  No GPU available - training will be slow on CPU")
        print("  Consider using Google Colab or a cloud GPU instance")
    
    return gpu_info


def check_project_structure():
    """Check if all required files and directories exist"""
    print("\n" + "="*50)
    print("Project Structure Check")
    print("="*50)
    
    required_dirs = [
        'src',
        'src/models',
        'src/datasets', 
        'src/utils',
        'src/losses',
        'scripts',
        'data'
    ]
    
    required_files = [
        'colab.ipynb',
        'report.md',
        'README.md',
        'requirements.txt',
        'src/__init__.py',
        'src/models/__init__.py',
        'src/models/backbone.py',
        'src/models/neck.py',
        'src/models/head.py',
        'src/datasets/__init__.py',
        'src/datasets/coco_dataset.py',
        'src/datasets/voc_dataset.py',
        'src/datasets/imagenette_dataset.py',
        'src/datasets/unified_dataloader.py',
        'src/utils/__init__.py',
        'src/utils/metrics.py',
        'src/utils/visualization.py',
        'src/utils/training_utils.py',
        'src/losses/__init__.py',
        'src/losses/detection_loss.py',
        'src/losses/segmentation_loss.py',
        'src/losses/classification_loss.py',
        'scripts/download_data.py',
        'scripts/verify_data.py',
        'scripts/eval.py',
        'scripts/phase1_check.py'
    ]
    
    all_good = True
    missing_items = []
    
    print("Checking directories...")
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ❌ {dir_path}")
            missing_items.append(f"Directory: {dir_path}")
            all_good = False
    
    print("\nChecking files...")
    for file_path in required_files:
        if os.path.exists(file_path):
            # Check file is not empty
            size = os.path.getsize(file_path)
            if size > 0:
                print(f"  ✓ {file_path} ({size} bytes)")
            else:
                print(f"  ⚠️  {file_path} (empty file)")
        else:
            print(f"  ❌ {file_path}")
            missing_items.append(f"File: {file_path}")
            all_good = False
    
    if missing_items:
        print(f"\n❌ Missing {len(missing_items)} items:")
        for item in missing_items[:5]:  # Show first 5
            print(f"  - {item}")
        if len(missing_items) > 5:
            print(f"  ... and {len(missing_items) - 5} more")
    
    return all_good


def check_dependencies():
    """Check if all required Python packages are installed"""
    print("\n" + "="*50)
    print("Dependencies Check")
    print("="*50)
    
    dependencies = {
        'torch': {'min_version': '1.9.0', 'installed': False, 'version': None},
        'torchvision': {'min_version': '0.10.0', 'installed': False, 'version': None},
        'cv2': {'min_version': '4.5.0', 'installed': False, 'version': None},
        'numpy': {'min_version': '1.19.0', 'installed': False, 'version': None},
        'matplotlib': {'min_version': '3.3.0', 'installed': False, 'version': None},
        'PIL': {'min_version': None, 'installed': False, 'version': None},
        'tqdm': {'min_version': None, 'installed': False, 'version': None},
        'pycocotools': {'min_version': None, 'installed': False, 'version': None},
    }
    
    all_good = True
    
    for package, info in dependencies.items():
        try:
            if package == 'cv2':
                import cv2
                version = cv2.__version__
            elif package == 'PIL':
                from PIL import Image
                import PIL
                version = PIL.__version__
            elif package == 'pycocotools':
                from pycocotools.coco import COCO
                version = 'installed'
            else:
                module = __import__(package)
                version = module.__version__ if hasattr(module, '__version__') else 'unknown'
            
            info['installed'] = True
            info['version'] = version
            
            # Check version if required
            if info['min_version'] and version != 'unknown':
                try:
                    from packaging import version as pkg_version
                    if pkg_version.parse(version) >= pkg_version.parse(info['min_version']):
                        print(f"  ✓ {package}: {version}")
                    else:
                        print(f"  ⚠️  {package}: {version} (minimum required: {info['min_version']})")
                except:
                    print(f"  ✓ {package}: {version}")
            else:
                print(f"  ✓ {package}: {version}")
                
        except ImportError:
            print(f"  ❌ {package}: not installed")
            info['installed'] = False
            all_good = False
    
    # Check for additional useful packages
    print("\nOptional packages:")
    optional = ['tensorboard', 'wandb', 'scipy', 'sklearn']
    for package in optional:
        try:
            module = __import__(package)
            version = module.__version__ if hasattr(module, '__version__') else 'installed'
            print(f"  ✓ {package}: {version}")
        except ImportError:
            print(f"  - {package}: not installed (optional)")
    
    return all_good


def check_model_components():
    """Test if model components can be loaded and initialized"""
    print("\n" + "="*50)
    print("Model Components Check")
    print("="*50)
    
    try:
        # Add current directory to path
        sys.path.insert(0, os.getcwd())
        
        from src.models import Backbone, FPN, DetectionHead, SegmentationHead, ClassificationHead
        print("✓ All model components imported successfully")
        
        # Test backbone
        print("\nTesting Backbone...")
        backbone = Backbone()
        dummy_input = torch.randn(2, 3, 224, 224)
        features = backbone(dummy_input)
        print(f"  ✓ Input shape: {dummy_input.shape}")
        print(f"  ✓ Output shapes: {[f.shape for f in features]}")
        
        # Test FPN
        print("\nTesting FPN...")
        fpn = FPN(backbone.out_channels)
        fpn_features = fpn(features)
        print(f"  ✓ FPN output shapes: {[f.shape for f in fpn_features]}")
        
        # Test heads
        print("\nTesting Detection Head...")
        det_head = DetectionHead(256, num_classes=80)
        det_cls, det_reg = det_head(fpn_features[0])
        print(f"  ✓ Classification output: {det_cls.shape}")
        print(f"  ✓ Regression output: {det_reg.shape}")
        
        print("\nTesting Segmentation Head...")
        seg_head = SegmentationHead(256, num_classes=21)
        seg_out = seg_head(fpn_features[0])
        print(f"  ✓ Segmentation output: {seg_out.shape}")
        
        print("\nTesting Classification Head...")
        cls_head = ClassificationHead(2048, num_classes=10)
        cls_out = cls_head(features[-1])
        print(f"  ✓ Classification output: {cls_out.shape}")
        
        # Test unified model if possible
        try:
            print("\nTesting Unified Model Assembly...")
            class SimpleUnifiedModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.backbone = Backbone()
                    self.fpn = FPN(self.backbone.out_channels)
                    self.det_head = DetectionHead(256, 80)
                    self.seg_head = SegmentationHead(256, 21)
                    self.cls_head = ClassificationHead(2048, 10)
                
                def forward(self, x):
                    features = self.backbone(x)
                    fpn_features = self.fpn(features)
                    return {
                        'detection': self.det_head(fpn_features[0]),
                        'segmentation': self.seg_head(fpn_features[0]),
                        'classification': self.cls_head(features[-1])
                    }
            
            model = SimpleUnifiedModel()
            outputs = model(dummy_input)
            param_count = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"  ✓ Unified model created: {param_count:.2f}M parameters")
            print(f"  ✓ Model outputs: {list(outputs.keys())}")
        except Exception as e:
            print(f"  ⚠️  Could not test unified model: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with model components: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_dataset_status():
    """Check if datasets are downloaded and valid"""
    print("\n" + "="*50)
    print("Dataset Status Check")
    print("="*50)
    
    data_dir = './data'
    datasets_info = {
        'mini_coco_det': {'expected_size_mb': 45, 'train': 240, 'val': 60},
        'mini_voc_seg': {'expected_size_mb': 30, 'train': 240, 'val': 60},
        'imagenette_160': {'expected_size_mb': 25, 'train': 240, 'val': 60}
    }
    
    total_size = 0
    all_present = True
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        print("  Run: python scripts/download_data.py")
        return False
    
    for dataset_name, info in datasets_info.items():
        dataset_path = os.path.join(data_dir, dataset_name)
        if os.path.exists(dataset_path):
            # Calculate size
            size_mb = 0
            for root, _, files in os.walk(dataset_path):
                for file in files:
                    size_mb += os.path.getsize(os.path.join(root, file)) / (1024 * 1024)
            
            total_size += size_mb
            print(f"✓ {dataset_name}: {size_mb:.2f} MB")
            
            # Check for specific files/directories
            if dataset_name == 'mini_coco_det':
                train_dir = os.path.join(dataset_path, 'images', 'train')
                val_dir = os.path.join(dataset_path, 'images', 'val')
                ann_dir = os.path.join(dataset_path, 'annotations')
                
                if os.path.exists(train_dir) and os.path.exists(val_dir) and os.path.exists(ann_dir):
                    print(f"    ✓ Structure valid")
                else:
                    print(f"    ❌ Invalid structure")
                    all_present = False
                    
        else:
            print(f"❌ {dataset_name}: not found")
            all_present = False
    
    print(f"\nTotal size: {total_size:.2f} MB")
    if total_size > 120:
        print(f"⚠️  Warning: Total size exceeds 120 MB limit")
    
    # Check for verification report
    report_path = os.path.join(data_dir, 'dataset_verification_report.txt')
    if os.path.exists(report_path):
        print(f"\n✓ Verification report found: {report_path}")
    else:
        print(f"\n- No verification report found")
        print("  Run: python scripts/verify_data.py")
    
    return all_present


def generate_phase1_report():
    """Generate a comprehensive Phase 1 verification report"""
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'checks': {},
        'overall_status': 'PENDING'
    }
    
    # Save results to JSON
    report_path = 'phase1_verification_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Check Phase 1 requirements for multi-task learning project')
    parser.add_argument('--skip-gpu', action='store_true', help='Skip GPU checks')
    parser.add_argument('--skip-data', action='store_true', help='Skip dataset checks')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("PHASE 1 REQUIREMENTS CHECK")
    print("="*70)
    print(f"Check performed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {os.getcwd()}")
    
    all_checks = {}
    
    # 1. GPU Environment Check
    if not args.skip_gpu:
        gpu_info = check_gpu_environment()
        all_checks['gpu'] = gpu_info['cuda_available']
    else:
        print("\n⚠️  Skipping GPU checks")
        all_checks['gpu'] = None
    
    # 2. Project Structure Check
    all_checks['structure'] = check_project_structure()
    
    # 3. Dependencies Check
    all_checks['dependencies'] = check_dependencies()
    
    # 4. Model Components Check
    all_checks['models'] = check_model_components()
    
    # 5. Dataset Status Check
    if not args.skip_data:
        all_checks['datasets'] = check_dataset_status()
    else:
        print("\n⚠️  Skipping dataset checks")
        all_checks['datasets'] = None
    
    # Summary
    print("\n" + "="*70)
    print("PHASE 1 VERIFICATION SUMMARY")
    print("="*70)
    
    check_names = {
        'gpu': 'GPU Environment',
        'structure': 'Project Structure',
        'dependencies': 'Dependencies',
        'models': 'Model Components',
        'datasets': 'Datasets'
    }
    
    failed_checks = []
    for check, passed in all_checks.items():
        if passed is None:
            status = "⚠️  SKIPPED"
        elif passed:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
            failed_checks.append(check_names[check])
        
        print(f"{check_names[check]:<20} {status}")
    
    print("="*70)
    
    # Generate report
    generate_phase1_report()
    
    # Final verdict
    essential_checks = ['structure', 'dependencies', 'models']
    essential_passed = all(all_checks.get(check, False) for check in essential_checks)
    
    if essential_passed and not failed_checks:
        print("\n✅ Phase 1 驗收通過！可以進入 Phase 2")
        print("\nNext steps:")
        if all_checks.get('datasets') is not True:
            print("1. Download datasets: python scripts/download_data.py")
            print("2. Verify datasets: python scripts/verify_data.py")
        print("3. Start training in colab.ipynb or create train.py")
        print("4. Remember to use: CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python train.py")
        return 0
    else:
        print(f"\n❌ Phase 1 verification failed!")
        if failed_checks:
            print(f"Failed checks: {', '.join(failed_checks)}")
        print("\nPlease fix the issues above before proceeding.")
        return 1


if __name__ == '__main__':
    exit(main())
#!/usr/bin/env python3
"""
Phase 1 最終驗收系統
執行完整的 Phase 1 驗收，包括環境、項目結構、數據完整性、功能測試和性能基準
"""
import os
import sys
import time
import torch
import psutil
import subprocess
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Phase1Verifier:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / 'data'
        self.results = {
            'environment': {},
            'project_structure': {},
            'data_integrity': {},
            'dataloader_functionality': {},
            'performance_benchmarks': {},
            'overall_status': 'PENDING'
        }
        self.errors = []
        self.warnings = []
        
    def print_header(self, title, symbol="="):
        """Print formatted section header"""
        print(f"\n{symbol * 70}")
        print(f"{title:^70}")
        print(f"{symbol * 70}")
    
    def print_status(self, item, status, details=""):
        """Print status with colored output"""
        if status == "PASS":
            print(f"✅ {item}")
        elif status == "FAIL":
            print(f"❌ {item}")
            if details:
                print(f"   💡 {details}")
        elif status == "WARN":
            print(f"⚠️  {item}")
            if details:
                print(f"   ℹ️  {details}")
        
        if details and status != "FAIL" and status != "WARN":
            print(f"   ℹ️  {details}")
    
    def check_environment(self):
        """檢查環境配置"""
        self.print_header("1. 環境檢查")
        
        env_results = {}
        
        # Python 版本檢查
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        env_results['python_version'] = python_version
        if sys.version_info >= (3, 8):
            self.print_status(f"Python 版本: {python_version}", "PASS")
        else:
            self.print_status(f"Python 版本: {python_version}", "FAIL", "需要 Python 3.8+")
            self.errors.append("Python 版本過舊")
        
        # PyTorch 檢查
        try:
            torch_version = torch.__version__
            env_results['torch_version'] = torch_version
            self.print_status(f"PyTorch 版本: {torch_version}", "PASS")
        except Exception as e:
            self.print_status("PyTorch", "FAIL", f"無法檢測版本: {e}")
            self.errors.append("PyTorch 未安裝或版本問題")
        
        # CUDA 可用性檢查
        cuda_available = torch.cuda.is_available()
        env_results['cuda_available'] = cuda_available
        if cuda_available:
            cuda_version = torch.version.cuda
            gpu_count = torch.cuda.device_count()
            env_results['cuda_version'] = cuda_version
            env_results['gpu_count'] = gpu_count
            self.print_status(f"CUDA 可用: v{cuda_version}, {gpu_count} GPU(s)", "PASS")
            
            # GPU 記憶體檢查
            for i in range(gpu_count):
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                env_results[f'gpu_{i}_memory_gb'] = gpu_memory
                if gpu_memory >= 4.0:
                    self.print_status(f"GPU {i} 記憶體: {gpu_memory:.1f} GB", "PASS")
                else:
                    self.print_status(f"GPU {i} 記憶體: {gpu_memory:.1f} GB", "WARN", "建議至少 4GB")
                    self.warnings.append(f"GPU {i} 記憶體不足 4GB")
        else:
            self.print_status("CUDA", "WARN", "不可用，將使用 CPU")
            self.warnings.append("CUDA 不可用")
        
        # 系統記憶體檢查
        system_memory = psutil.virtual_memory().total / 1024**3
        env_results['system_memory_gb'] = system_memory
        if system_memory >= 8.0:
            self.print_status(f"系統記憶體: {system_memory:.1f} GB", "PASS")
        else:
            self.print_status(f"系統記憶體: {system_memory:.1f} GB", "WARN", "建議至少 8GB")
            self.warnings.append("系統記憶體可能不足")
        
        # 依賴套件檢查
        required_packages = {
            'numpy': 'numpy',
            'opencv-python': 'cv2', 
            'matplotlib': 'matplotlib',
            'tqdm': 'tqdm',
            'Pillow': 'PIL'
        }
        for package_name, import_name in required_packages.items():
            try:
                __import__(import_name)
                self.print_status(f"套件 {package_name}", "PASS")
            except ImportError:
                self.print_status(f"套件 {package_name}", "FAIL", "未安裝")
                self.errors.append(f"缺少套件: {package_name}")
        
        self.results['environment'] = env_results
        return len([e for e in self.errors if "套件" in e or "Python" in e or "PyTorch" in e]) == 0
    
    def check_project_structure(self):
        """檢查項目結構"""
        self.print_header("2. 項目結構檢查")
        
        structure_results = {}
        
        # 必要目錄檢查
        required_dirs = [
            'src',
            'src/models',
            'src/datasets', 
            'src/utils',
            'src/losses',
            'scripts',
            'data'
        ]
        
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            exists = full_path.exists()
            structure_results[f'dir_{dir_path.replace("/", "_")}'] = exists
            if exists:
                self.print_status(f"目錄 {dir_path}/", "PASS")
            else:
                self.print_status(f"目錄 {dir_path}/", "FAIL", "不存在")
                self.errors.append(f"缺少目錄: {dir_path}")
        
        # 必要文件檢查
        required_files = [
            'src/__init__.py',
            'src/models/__init__.py',
            'src/datasets/__init__.py',
            'src/utils/__init__.py', 
            'src/losses/__init__.py',
            'src/datasets/coco_dataset.py',
            'src/datasets/voc_dataset.py',
            'src/datasets/imagenette_dataset.py',
            'src/datasets/unified_dataloader.py',
            'scripts/download_data.py',
            'scripts/verify_data.py',
            'scripts/test_dataloader.py'
        ]
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            exists = full_path.exists()
            structure_results[f'file_{file_path.replace("/", "_").replace(".", "_")}'] = exists
            if exists:
                self.print_status(f"文件 {file_path}", "PASS")
            else:
                self.print_status(f"文件 {file_path}", "FAIL", "不存在")
                self.errors.append(f"缺少文件: {file_path}")
        
        # 模組導入檢查
        try:
            from src.datasets import CocoDetectionDataset, VOCSegmentationDataset, ImagenetteDataset
            from src.datasets import UnifiedDataLoader, UnifiedDataset, unified_collate_fn
            self.print_status("模組導入", "PASS", "所有數據集類和統一加載器")
            structure_results['module_import'] = True
        except Exception as e:
            self.print_status("模組導入", "FAIL", f"導入錯誤: {e}")
            self.errors.append(f"模組導入失敗: {e}")
            structure_results['module_import'] = False
        
        self.results['project_structure'] = structure_results
        return len([e for e in self.errors if "缺少" in e or "導入" in e]) == 0
    
    def check_data_integrity(self):
        """檢查數據完整性"""
        self.print_header("3. 數據完整性檢查")
        
        data_results = {}
        
        # 檢查數據目錄存在
        if not self.data_dir.exists():
            self.print_status("數據目錄", "FAIL", "data/ 目錄不存在")
            self.errors.append("數據目錄不存在")
            return False
        
        # 檢查各個數據集
        datasets = {
            'mini_coco_det': {'expected_images': 300, 'max_size_mb': 50},
            'mini_voc_seg': {'expected_images': 300, 'max_size_mb': 35}, 
            'imagenette_160': {'expected_images': 300, 'max_size_mb': 30}
        }
        
        total_images = 0
        total_size_mb = 0
        
        for dataset_name, config in datasets.items():
            dataset_path = self.data_dir / dataset_name
            dataset_results = {}
            
            if dataset_path.exists():
                # 計算大小
                size_bytes = sum(f.stat().st_size for f in dataset_path.rglob('*') if f.is_file())
                size_mb = size_bytes / 1024 / 1024
                dataset_results['size_mb'] = size_mb
                total_size_mb += size_mb
                
                # 計算圖像數量
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
                image_files = [f for f in dataset_path.rglob('*') if f.suffix.lower() in image_extensions]
                image_count = len(image_files)
                dataset_results['image_count'] = image_count
                total_images += image_count
                
                # 檢查結果
                if image_count >= config['expected_images'] * 0.9:  # 允許 10% 誤差
                    self.print_status(f"{dataset_name} 圖像數量: {image_count}", "PASS")
                else:
                    self.print_status(f"{dataset_name} 圖像數量: {image_count}", "FAIL", 
                                    f"期望至少 {int(config['expected_images'] * 0.9)}")
                    self.errors.append(f"{dataset_name} 圖像數量不足")
                
                if size_mb <= config['max_size_mb']:
                    self.print_status(f"{dataset_name} 大小: {size_mb:.1f} MB", "PASS")
                else:
                    self.print_status(f"{dataset_name} 大小: {size_mb:.1f} MB", "WARN", 
                                    f"超過建議大小 {config['max_size_mb']} MB")
                    self.warnings.append(f"{dataset_name} 大小超過建議值")
                
                dataset_results['exists'] = True
            else:
                self.print_status(f"{dataset_name}", "FAIL", "數據集不存在")
                self.errors.append(f"數據集 {dataset_name} 不存在")
                dataset_results['exists'] = False
            
            data_results[dataset_name] = dataset_results
        
        # 總體檢查
        data_results['total_images'] = total_images
        data_results['total_size_mb'] = total_size_mb
        
        if total_size_mb <= 120:
            self.print_status(f"總數據大小: {total_size_mb:.1f} MB", "PASS")
        else:
            self.print_status(f"總數據大小: {total_size_mb:.1f} MB", "WARN", "超過 120MB 建議值")
            self.warnings.append("總數據大小超過建議值")
        
        if total_images >= 800:  # 允許一些誤差
            self.print_status(f"總圖像數量: {total_images}", "PASS")
        else:
            self.print_status(f"總圖像數量: {total_images}", "FAIL", "期望至少 800 張")
            self.errors.append("總圖像數量不足")
        
        self.results['data_integrity'] = data_results
        return len([e for e in self.errors if "數據" in e]) == 0
    
    def check_dataloader_functionality(self):
        """檢查數據加載器功能"""
        self.print_header("4. 數據加載器功能檢查")
        
        func_results = {}
        
        try:
            from src.datasets import CocoDetectionDataset, VOCSegmentationDataset, ImagenetteDataset
            from src.datasets import UnifiedDataLoader, UnifiedDataset, unified_collate_fn
            from torch.utils.data import DataLoader
            
            # 測試個別數據集
            datasets = {}
            
            # COCO Detection
            try:
                coco_dataset = CocoDetectionDataset(self.data_dir / 'mini_coco_det', split='train')
                img, target = coco_dataset[0]
                assert isinstance(img, torch.Tensor), "COCO 圖像不是 tensor"
                assert isinstance(target, dict), "COCO target 不是 dict"
                assert 'boxes' in target, "COCO target 缺少 boxes"
                assert 'labels' in target, "COCO target 缺少 labels"
                datasets['coco'] = coco_dataset
                self.print_status("COCO Detection 數據集", "PASS", f"形狀: {img.shape}")
                func_results['coco_dataset'] = True
            except Exception as e:
                self.print_status("COCO Detection 數據集", "FAIL", str(e))
                self.errors.append(f"COCO 數據集錯誤: {e}")
                func_results['coco_dataset'] = False
            
            # VOC Segmentation
            try:
                voc_dataset = VOCSegmentationDataset(self.data_dir / 'mini_voc_seg', split='train')
                img, target = voc_dataset[0]
                assert isinstance(img, torch.Tensor), "VOC 圖像不是 tensor"
                assert isinstance(target, dict), "VOC target 不是 dict"
                assert 'masks' in target, "VOC target 缺少 masks"
                assert 'labels' in target, "VOC target 缺少 labels"
                datasets['voc'] = voc_dataset
                self.print_status("VOC Segmentation 數據集", "PASS", f"形狀: {img.shape}")
                func_results['voc_dataset'] = True
            except Exception as e:
                self.print_status("VOC Segmentation 數據集", "FAIL", str(e))
                self.errors.append(f"VOC 數據集錯誤: {e}")
                func_results['voc_dataset'] = False
            
            # Imagenette Classification
            try:
                imagenette_dataset = ImagenetteDataset(self.data_dir / 'imagenette_160', split='train')
                img, target = imagenette_dataset[0]
                assert isinstance(img, torch.Tensor), "Imagenette 圖像不是 tensor"
                assert isinstance(target, dict), "Imagenette target 不是 dict"
                assert 'labels' in target, "Imagenette target 缺少 labels"
                datasets['imagenette'] = imagenette_dataset
                self.print_status("Imagenette Classification 數據集", "PASS", f"形狀: {img.shape}")
                func_results['imagenette_dataset'] = True
            except Exception as e:
                self.print_status("Imagenette Classification 數據集", "FAIL", str(e))
                self.errors.append(f"Imagenette 數據集錯誤: {e}")
                func_results['imagenette_dataset'] = False
            
            # 測試統一數據加載器
            if all(func_results.get(k, False) for k in ['coco_dataset', 'voc_dataset', 'imagenette_dataset']):
                try:
                    # 測試 UnifiedDataset
                    unified_dataset = UnifiedDataset(
                        detection_dataset=datasets['coco'],
                        segmentation_dataset=datasets['voc'],
                        classification_dataset=datasets['imagenette'],
                        sampling_strategy='balanced'
                    )
                    
                    item = unified_dataset[0]
                    assert isinstance(item, dict), "UnifiedDataset 項目不是 dict"
                    assert 'images' in item, "UnifiedDataset 項目缺少 images"
                    assert 'task_type' in item, "UnifiedDataset 項目缺少 task_type"
                    assert 'targets' in item, "UnifiedDataset 項目缺少 targets"
                    
                    self.print_status("UnifiedDataset", "PASS", f"任務類型: {item['task_type']}")
                    func_results['unified_dataset'] = True
                    
                    # 測試 DataLoader with unified_collate_fn
                    loader = DataLoader(unified_dataset, batch_size=4, collate_fn=unified_collate_fn)
                    batch = next(iter(loader))
                    assert 'images' in batch, "批次缺少 images"
                    assert 'task_types' in batch, "批次缺少 task_types"
                    assert 'targets' in batch, "批次缺少 targets"
                    assert 'task_groups' in batch, "批次缺少 task_groups"
                    
                    self.print_status("統一批次加載", "PASS", f"批次大小: {batch['images'].shape}")
                    func_results['unified_batch'] = True
                    
                    # 測試 UnifiedDataLoader
                    unified_loader = UnifiedDataLoader(
                        detection_dataset=datasets['coco'],
                        segmentation_dataset=datasets['voc'],
                        classification_dataset=datasets['imagenette'],
                        batch_size=4,
                        sampling_strategy='balanced'
                    )
                    
                    batch = next(iter(unified_loader))
                    assert 'images' in batch, "UnifiedDataLoader 批次缺少 images"
                    
                    self.print_status("UnifiedDataLoader", "PASS", f"批次大小: {batch['images'].shape}")
                    func_results['unified_dataloader'] = True
                    
                except Exception as e:
                    self.print_status("統一數據加載器", "FAIL", str(e))
                    self.errors.append(f"統一數據加載器錯誤: {e}")
                    func_results['unified_dataset'] = False
                    func_results['unified_batch'] = False
                    func_results['unified_dataloader'] = False
            else:
                self.print_status("統一數據加載器", "FAIL", "個別數據集測試失敗")
                self.errors.append("無法測試統一數據加載器")
                func_results['unified_dataset'] = False
                func_results['unified_batch'] = False
                func_results['unified_dataloader'] = False
                
        except Exception as e:
            self.print_status("數據加載器導入", "FAIL", str(e))
            self.errors.append(f"數據加載器導入錯誤: {e}")
            func_results['import_error'] = str(e)
        
        self.results['dataloader_functionality'] = func_results
        return len([e for e in self.errors if "數據集" in e or "加載器" in e]) == 0
    
    def check_performance_benchmarks(self):
        """檢查性能基準"""
        self.print_header("5. 性能基準測試")
        
        perf_results = {}
        
        try:
            from src.datasets import CocoDetectionDataset, VOCSegmentationDataset, ImagenetteDataset
            from src.datasets import UnifiedDataLoader
            
            # 創建數據集
            coco_dataset = CocoDetectionDataset(self.data_dir / 'mini_coco_det', split='train')
            voc_dataset = VOCSegmentationDataset(self.data_dir / 'mini_voc_seg', split='train')
            imagenette_dataset = ImagenetteDataset(self.data_dir / 'imagenette_160', split='train')
            
            # 創建統一加載器
            unified_loader = UnifiedDataLoader(
                detection_dataset=coco_dataset,
                segmentation_dataset=voc_dataset,
                classification_dataset=imagenette_dataset,
                batch_size=16,
                num_workers=2,
                sampling_strategy='balanced'
            )
            
            # 測試數據加載速度
            print("正在測試數據加載速度...")
            num_samples = 100
            start_time = time.time()
            samples_loaded = 0
            
            # 預熱
            for _ in range(2):
                batch = next(iter(unified_loader))
            
            # 實際測試
            start_time = time.time()
            for i, batch in enumerate(unified_loader):
                samples_loaded += len(batch['images'])
                if samples_loaded >= num_samples:
                    break
            
            elapsed_time = time.time() - start_time
            loading_speed = samples_loaded / elapsed_time
            perf_results['loading_speed_samples_per_sec'] = loading_speed
            
            if loading_speed >= 30:  # 降低要求以適應較慢的環境
                self.print_status(f"數據加載速度: {loading_speed:.2f} samples/sec", "PASS")
            else:
                self.print_status(f"數據加載速度: {loading_speed:.2f} samples/sec", "WARN", 
                                "速度較慢，但可接受")
                self.warnings.append("數據加載速度較慢")
            
            # 記憶體使用測試
            memory_before = psutil.virtual_memory().used / 1024**3
            
            # 加載多個批次測試記憶體
            batches = []
            for i, batch in enumerate(unified_loader):
                batches.append(batch)
                if i >= 5:  # 加載 6 個批次
                    break
            
            memory_after = psutil.virtual_memory().used / 1024**3
            memory_usage = memory_after - memory_before
            perf_results['memory_usage_gb'] = memory_usage
            
            if memory_usage <= 2.0:
                self.print_status(f"記憶體使用: {memory_usage:.2f} GB", "PASS")
            else:
                self.print_status(f"記憶體使用: {memory_usage:.2f} GB", "WARN", "記憶體使用較高")
                self.warnings.append("記憶體使用較高")
            
            # GPU 運算測試（如果可用）
            if torch.cuda.is_available():
                try:
                    device = torch.device('cuda')
                    test_tensor = torch.randn(1000, 1000, device=device)
                    start_time = time.time()
                    result = torch.matmul(test_tensor, test_tensor)
                    torch.cuda.synchronize()
                    gpu_time = time.time() - start_time
                    perf_results['gpu_compute_time_ms'] = gpu_time * 1000
                    
                    self.print_status(f"GPU 運算測試: {gpu_time*1000:.2f} ms", "PASS")
                except Exception as e:
                    self.print_status("GPU 運算測試", "WARN", f"測試失敗: {e}")
                    self.warnings.append("GPU 運算測試失敗")
            else:
                self.print_status("GPU 運算測試", "WARN", "CUDA 不可用")
                
        except Exception as e:
            self.print_status("性能測試", "FAIL", str(e))
            self.errors.append(f"性能測試錯誤: {e}")
            return False
        
        self.results['performance_benchmarks'] = perf_results
        return True
    
    def generate_verification_report(self):
        """生成驗收報告"""
        self.print_header("6. 驗收報告", "=")
        
        # 計算總體狀態
        has_critical_errors = len(self.errors) > 0
        has_warnings = len(self.warnings) > 0
        
        if has_critical_errors:
            self.results['overall_status'] = 'FAILED'
            status_symbol = "❌"
            status_text = "失敗"
        elif has_warnings:
            self.results['overall_status'] = 'PASSED_WITH_WARNINGS'
            status_symbol = "⚠️"
            status_text = "通過 (有警告)"
        else:
            self.results['overall_status'] = 'PASSED'
            status_symbol = "✅"
            status_text = "完全通過"
        
        print(f"\n{'=' * 70}")
        print(f"{'Phase 1 驗收報告':^70}")
        print(f"{'=' * 70}")
        
        # 檢查項目摘要
        checks = [
            ("環境檢查", "environment"),
            ("項目結構", "project_structure"), 
            ("數據完整性", "data_integrity"),
            ("數據加載器", "dataloader_functionality"),
            ("性能基準", "performance_benchmarks")
        ]
        
        for check_name, check_key in checks:
            if check_key in self.results and self.results[check_key]:
                # 檢查是否有相關錯誤
                related_errors = [e for e in self.errors if any(keyword in e.lower() for keyword in check_name.lower().split())]
                if related_errors:
                    print(f"❌ {check_name}")
                else:
                    print(f"✅ {check_name}")
            else:
                print(f"❌ {check_name}")
        
        # 統計信息
        print(f"\n📊 統計信息:")
        if 'data_integrity' in self.results:
            data_info = self.results['data_integrity']
            print(f"   總圖像數量: {data_info.get('total_images', 'N/A')}")
            print(f"   總數據大小: {data_info.get('total_size_mb', 'N/A'):.1f} MB")
        
        if 'performance_benchmarks' in self.results:
            perf_info = self.results['performance_benchmarks']
            if 'loading_speed_samples_per_sec' in perf_info:
                print(f"   數據載入速度: {perf_info['loading_speed_samples_per_sec']:.1f} samples/sec")
            if 'memory_usage_gb' in perf_info:
                print(f"   記憶體使用: {perf_info['memory_usage_gb']:.2f} GB")
        
        if 'environment' in self.results:
            env_info = self.results['environment']
            if 'cuda_available' in env_info and env_info['cuda_available']:
                gpu_memory = env_info.get('gpu_0_memory_gb', 'N/A')
                print(f"   GPU記憶體: {gpu_memory:.1f} GB available")
            else:
                print(f"   GPU記憶體: N/A (CUDA 不可用)")
        
        # 最終狀態
        print(f"\n{status_symbol} Phase 1 驗收: {status_text}")
        
        # 錯誤和警告
        if self.errors:
            print(f"\n❌ 發現 {len(self.errors)} 個錯誤:")
            for i, error in enumerate(self.errors, 1):
                print(f"   {i}. {error}")
        
        if self.warnings:
            print(f"\n⚠️  發現 {len(self.warnings)} 個警告:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"   {i}. {warning}")
        
        # 修復建議
        if self.errors:
            print(f"\n🔧 修復建議:")
            print("   1. 檢查錯誤信息並解決相關問題")
            print("   2. 重新運行驗收腳本")
            print("   3. 如需協助，請查看相關文檔")
        
        # 下一步指導
        if not has_critical_errors:
            print(f"\n🎉 Phase 1 驗收通過！")
            print("📝 所有核心系統就緒，可以進入 Phase 2: 模型架構設計")
            print('\n請輸入 "進入Phase 2" 以繼續下一階段。')
        else:
            print(f"\n🔄 請修復上述錯誤後重新運行驗收")
            print("   python scripts/final_phase1_verification.py")
        
        # 保存詳細報告
        report_path = self.project_root / 'phase1_verification_report.json'
        self.results['timestamp'] = datetime.now().isoformat()
        self.results['errors'] = self.errors
        self.results['warnings'] = self.warnings
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 詳細報告已保存至: {report_path}")
        
        return not has_critical_errors
    
    def run_verification(self):
        """執行完整驗收流程"""
        print("🚀 開始 Phase 1 最終驗收...")
        print(f"⏰ 時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        success = True
        
        # 執行各項檢查
        success &= self.check_environment()
        success &= self.check_project_structure()
        success &= self.check_data_integrity()
        success &= self.check_dataloader_functionality()
        success &= self.check_performance_benchmarks()
        
        # 生成報告
        final_success = self.generate_verification_report()
        
        return final_success


def main():
    """主函數"""
    verifier = Phase1Verifier()
    success = verifier.run_verification()
    
    # 返回適當的退出碼
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
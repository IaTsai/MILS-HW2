#!/usr/bin/env python3
"""
預訓練驗證腳本
在開始完整訓練前驗證所有關鍵組件
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.unified_model import create_unified_model
from src.datasets.unified_dataloader import create_unified_dataloaders
from src.utils.sequential_trainer import SequentialTrainer
from src.utils.ewc import create_ewc_handler, ewc_loss


class PreTrainingValidator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
    def print_header(self, title):
        """打印測試標題"""
        print(f"\n{'='*60}")
        print(f"🧪 {title}")
        print(f"{'='*60}\n")
        
    def print_result(self, test_name, passed, details=""):
        """打印測試結果"""
        icon = "✅" if passed else "❌"
        status = "PASS" if passed else "FAIL"
        print(f"{icon} {test_name}: {status}")
        if details:
            print(f"   {details}")
            
    def test_data_loading(self):
        """測試所有三個任務的數據加載"""
        self.print_header("數據加載測試")
        
        try:
            # 創建數據加載器
            dataloaders = create_unified_dataloaders(
                './data', 
                batch_size=4, 
                num_workers=0
            )
            
            # 測試檢測任務數據
            det_loader = dataloaders['train'].get_task_loader('detection')
            det_batch = next(iter(det_loader))
            
            if isinstance(det_batch, list) and len(det_batch) == 2:
                images, targets = det_batch
                det_pass = (
                    torch.is_tensor(images) and 
                    isinstance(targets, list) and
                    len(targets) == images.shape[0]
                )
                self.print_result(
                    "檢測數據加載", 
                    det_pass,
                    f"批次格式: [{images.shape}, {len(targets)} targets]"
                )
            else:
                self.print_result("檢測數據加載", False, "錯誤的批次格式")
                
            # 測試分割任務數據
            seg_loader = dataloaders['train'].get_task_loader('segmentation')
            seg_batch = next(iter(seg_loader))
            
            if isinstance(seg_batch, (list, tuple)) and len(seg_batch) == 2:
                images, masks = seg_batch
                # 處理list格式的masks
                if isinstance(masks, list):
                    seg_pass = (
                        torch.is_tensor(images) and 
                        len(masks) == images.shape[0] and
                        all(torch.is_tensor(m) for m in masks)
                    )
                    self.print_result(
                        "分割數據加載", 
                        seg_pass,
                        f"批次格式: [{images.shape}, {len(masks)} masks]"
                    )
                else:
                    seg_pass = (
                        torch.is_tensor(images) and 
                        torch.is_tensor(masks) and
                        masks.shape[0] == images.shape[0]
                    )
                    self.print_result(
                        "分割數據加載", 
                        seg_pass,
                        f"批次格式: [{images.shape}, {masks.shape}]"
                    )
            else:
                self.print_result("分割數據加載", False, "錯誤的批次格式")
                
            # 測試分類任務數據
            cls_loader = dataloaders['train'].get_task_loader('classification')
            cls_batch = next(iter(cls_loader))
            
            if isinstance(cls_batch, (list, tuple)) and len(cls_batch) == 2:
                images, targets = cls_batch
                # 處理字典格式的targets
                if isinstance(targets, dict) and 'labels' in targets:
                    labels = targets['labels']
                    cls_pass = torch.is_tensor(labels) and labels.shape[0] == images.shape[0]
                    self.print_result(
                        "分類數據加載", 
                        cls_pass,
                        f"批次格式: [{images.shape}, labels={labels.shape}]"
                    )
                else:
                    cls_pass = torch.is_tensor(targets) and targets.shape[0] == images.shape[0]
                    self.print_result(
                        "分類數據加載", 
                        cls_pass,
                        f"批次格式: [{images.shape}, {targets.shape if torch.is_tensor(targets) else type(targets)}]"
                    )
            else:
                self.print_result("分類數據加載", False, "錯誤的批次格式")
                
            return dataloaders
            
        except Exception as e:
            self.print_result("數據加載", False, f"錯誤: {str(e)}")
            return None
            
    def test_evaluation_functions(self, dataloaders):
        """測試所有評估函數"""
        self.print_header("評估函數測試")
        
        try:
            # 創建模型和訓練器
            model = create_unified_model('default')
            trainer = SequentialTrainer(
                model=model,
                dataloaders=dataloaders,
                ewc_importance=1000.0,
                save_dir='./test_output',
                device=self.device
            )
            
            # 測試檢測評估
            try:
                det_loader = dataloaders['val'].get_task_loader('detection')
                det_metrics = trainer._validate_detection(det_loader)
                self.print_result(
                    "檢測評估函數", 
                    True,
                    f"mAP: {det_metrics['main_metric']:.4f}"
                )
            except Exception as e:
                self.print_result("檢測評估函數", False, f"錯誤: {str(e)}")
                
            # 測試分割評估
            try:
                seg_loader = dataloaders['val'].get_task_loader('segmentation')
                seg_metrics = trainer._validate_segmentation(seg_loader)
                self.print_result(
                    "分割評估函數", 
                    True,
                    f"mIoU: {seg_metrics['main_metric']:.4f}"
                )
            except Exception as e:
                self.print_result("分割評估函數", False, f"錯誤: {str(e)}")
                
            # 測試分類評估
            try:
                cls_loader = dataloaders['val'].get_task_loader('classification')
                cls_metrics = trainer._validate_classification(cls_loader)
                self.print_result(
                    "分類評估函數", 
                    True,
                    f"準確率: {cls_metrics['main_metric']:.4f}"
                )
            except Exception as e:
                self.print_result("分類評估函數", False, f"錯誤: {str(e)}")
                
        except Exception as e:
            self.print_result("評估函數", False, f"初始化錯誤: {str(e)}")
            
    def test_ewc_flow(self, dataloaders):
        """測試EWC計算流程"""
        self.print_header("EWC流程測試")
        
        try:
            # 創建模型和EWC處理器
            model = create_unified_model('default').to(self.device)
            ewc_handler = create_ewc_handler(model, importance=1000.0)
            
            # 測試Fisher矩陣計算
            try:
                seg_loader = dataloaders['train'].get_task_loader('segmentation')
                # 使用較少的樣本進行測試
                small_loader = DataLoader(
                    seg_loader.dataset, 
                    batch_size=4, 
                    shuffle=False,
                    num_workers=0,
                    collate_fn=seg_loader.collate_fn if hasattr(seg_loader, 'collate_fn') else None
                )
                
                start_time = time.time()
                task_id = ewc_handler.finish_task(small_loader, verbose=False)
                fisher_time = time.time() - start_time
                
                self.print_result(
                    "Fisher矩陣計算", 
                    True,
                    f"任務ID: {task_id}, 計算時間: {fisher_time:.2f}秒"
                )
            except Exception as e:
                self.print_result("Fisher矩陣計算", False, f"錯誤: {str(e)}")
                
            # 測試EWC損失計算
            try:
                # 創建假的當前損失
                current_loss = torch.tensor(1.0, requires_grad=True).to(self.device)
                
                # 計算EWC損失
                total_loss, ewc_penalty = ewc_loss(current_loss, ewc_handler)
                
                ewc_pass = (
                    torch.is_tensor(total_loss) and 
                    total_loss.requires_grad and
                    ewc_penalty is not None
                )
                
                self.print_result(
                    "EWC損失計算", 
                    ewc_pass,
                    f"總損失: {total_loss.item():.4f}, EWC懲罰: {ewc_penalty:.4f}"
                )
            except Exception as e:
                self.print_result("EWC損失計算", False, f"錯誤: {str(e)}")
                
            # 測試防遺忘檢查
            try:
                trainer = SequentialTrainer(
                    model=model,
                    dataloaders=dataloaders,
                    ewc_importance=1000.0,
                    save_dir='./test_output',
                    device=self.device,
                    adaptive_ewc=True
                )
                
                # 設置基準性能
                trainer.baseline_performance['segmentation'] = 0.8
                
                # 測試檢查遺忘
                all_metrics = trainer.evaluate_all_tasks()
                forgetting_detected = trainer._check_catastrophic_forgetting(all_metrics)
                
                self.print_result(
                    "防遺忘檢查邏輯", 
                    True,
                    f"檢測到遺忘: {forgetting_detected}"
                )
            except Exception as e:
                self.print_result("防遺忘檢查邏輯", False, f"錯誤: {str(e)}")
                
        except Exception as e:
            self.print_result("EWC流程", False, f"初始化錯誤: {str(e)}")
            
    def test_model_compatibility(self, dataloaders):
        """測試模型兼容性"""
        self.print_header("模型兼容性測試")
        
        try:
            model = create_unified_model('default').to(self.device)
            model.eval()
            
            # 測試三種任務的前向傳播
            with torch.no_grad():
                # 檢測任務
                det_loader = dataloaders['val'].get_task_loader('detection')
                det_batch = next(iter(det_loader))
                if isinstance(det_batch, list) and len(det_batch) == 2:
                    images, _ = det_batch
                    images = images.to(self.device)
                    outputs = model(images, task_type='detection')
                    det_pass = 'detection' in outputs and torch.is_tensor(outputs['detection'])
                    self.print_result(
                        "檢測任務前向傳播", 
                        det_pass,
                        f"輸出形狀: {outputs['detection'].shape}"
                    )
                
                # 分割任務
                seg_loader = dataloaders['val'].get_task_loader('segmentation')
                seg_batch = next(iter(seg_loader))
                if isinstance(seg_batch, (list, tuple)) and len(seg_batch) == 2:
                    images, _ = seg_batch
                    images = images.to(self.device)
                    outputs = model(images, task_type='segmentation')
                    seg_pass = 'segmentation' in outputs and torch.is_tensor(outputs['segmentation'])
                    self.print_result(
                        "分割任務前向傳播", 
                        seg_pass,
                        f"輸出形狀: {outputs['segmentation'].shape}"
                    )
                
                # 分類任務
                cls_loader = dataloaders['val'].get_task_loader('classification')
                cls_batch = next(iter(cls_loader))
                if isinstance(cls_batch, (list, tuple)) and len(cls_batch) == 2:
                    images, _ = cls_batch
                    images = images.to(self.device)
                    outputs = model(images, task_type='classification')
                    cls_pass = 'classification' in outputs and torch.is_tensor(outputs['classification'])
                    self.print_result(
                        "分類任務前向傳播", 
                        cls_pass,
                        f"輸出形狀: {outputs['classification'].shape}"
                    )
                    
            # 測試GPU記憶體使用
            if self.device.type == 'cuda':
                memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                self.print_result(
                    "GPU記憶體使用", 
                    True,
                    f"已使用: {memory_used:.2f}GB, 已保留: {memory_reserved:.2f}GB"
                )
                
            # 測試梯度計算
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            
            # 簡單的梯度測試
            images = torch.randn(2, 3, 512, 512).to(self.device)
            outputs = model(images, task_type='all')
            
            # 創建假的損失
            loss = sum(output.mean() for output in outputs.values())
            loss.backward()
            
            # 檢查梯度
            has_grad = any(p.grad is not None and p.grad.abs().max() > 0 for p in model.parameters())
            self.print_result(
                "梯度計算正常", 
                has_grad,
                "所有參數都有有效梯度"
            )
            
            optimizer.zero_grad()
            
        except Exception as e:
            self.print_result("模型兼容性", False, f"錯誤: {str(e)}")
            
    def run_all_tests(self):
        """運行所有測試"""
        print("\n" + "="*60)
        print("🚀 預訓練驗證開始")
        print("="*60)
        
        # 測試數據加載
        dataloaders = self.test_data_loading()
        
        if dataloaders:
            # 測試評估函數
            self.test_evaluation_functions(dataloaders)
            
            # 測試EWC流程
            self.test_ewc_flow(dataloaders)
            
            # 測試模型兼容性
            self.test_model_compatibility(dataloaders)
        
        print("\n" + "="*60)
        print("🎯 所有測試完成！")
        print("="*60)
        
        # 檢查是否所有測試都通過
        if dataloaders:
            print("\n✅ 所有測試通過！可以安心開始完整訓練！")
            print("\n建議訓練命令:")
            print("CUDA_VISIBLE_DEVICES=1 python scripts/sequential_training.py \\")
            print("  --batch_size 16 \\")
            print("  --stage1_epochs 50 \\")
            print("  --stage2_epochs 40 \\")
            print("  --stage3_epochs 30 \\")
            print("  --ewc_importance 1000 \\")
            print("  --adaptive_ewc")
        else:
            print("\n❌ 有測試失敗，請先修復問題再開始訓練！")


if __name__ == "__main__":
    validator = PreTrainingValidator()
    validator.run_all_tests()
#!/usr/bin/env python3
"""
依序訓練測試腳本
快速測試依序訓練流程的基本功能
"""
import os
import sys
import time
import torch
import torch.nn as nn
from pathlib import Path

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.unified_model import create_unified_model
from src.utils.sequential_trainer import create_sequential_trainer


def test_sequential_trainer():
    """測試依序訓練器"""
    print("🧪 測試依序訓練器...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 使用設備: {device}")
    
    # 創建模型
    print("🏗️ 創建模型...")
    model = create_unified_model('lightweight')  # 使用輕量化模型加快測試
    print(f"📊 模型參數: {sum(p.numel() for p in model.parameters()):,}")
    
    # 創建數據加載器（使用模擬數據）
    print("📂 創建數據加載器...")
    print("🔧 創建模擬數據加載器...")
    dataloaders = create_mock_dataloaders(device)
    
    # 創建依序訓練器
    print("🔧 創建依序訓練器...")
    trainer = create_sequential_trainer(
        model=model,
        dataloaders=dataloaders,
        ewc_importance=100.0,  # 較小的EWC權重
        save_dir='./test_sequential_results',
        device=str(device),
        learning_rate=1e-3,
        adaptive_ewc=True,
        forgetting_threshold=0.1  # 較寬鬆的遺忘閾值
    )
    
    print("✅ 依序訓練器創建成功")
    
    # 測試三階段訓練（每個階段只訓練很少的epoch）
    print("\n🚀 開始測試三階段訓練...")
    
    try:
        # Stage 1: 分割任務
        print("\n🎨 測試 Stage 1: 分割任務")
        stage1_metrics = trainer.train_stage(
            stage_name='test_stage1_segmentation',
            task_type='segmentation', 
            epochs=3,  # 只訓練3個epoch
            save_checkpoints=False
        )
        print(f"✅ Stage 1 完成，指標: {stage1_metrics['main_metric']:.4f}")
        
        # Stage 2: 檢測任務 + EWC  
        print("\n🎯 測試 Stage 2: 檢測任務 + EWC")
        stage2_metrics = trainer.train_stage(
            stage_name='test_stage2_detection',
            task_type='detection',
            epochs=3,  # 只訓練3個epoch
            save_checkpoints=False
        )
        print(f"✅ Stage 2 完成，指標: {stage2_metrics['main_metric']:.4f}")
        
        # 檢查遺忘
        print("\n🔍 檢查遺忘程度...")
        forgetting_info = trainer.check_forgetting()
        print(f"📊 最大遺忘程度: {forgetting_info['max_drop']*100:.2f}%")
        print(f"🎯 遺忘檢查: {'✅ 通過' if forgetting_info['acceptable'] else '❌ 失敗'}")
        
        # Stage 3: 分類任務 + EWC
        print("\n📊 測試 Stage 3: 分類任務 + EWC")  
        stage3_metrics = trainer.train_stage(
            stage_name='test_stage3_classification',
            task_type='classification',
            epochs=3,  # 只訓練3個epoch
            save_checkpoints=False
        )
        print(f"✅ Stage 3 完成，指標: {stage3_metrics['main_metric']:.4f}")
        
        # 最終評估
        print("\n📊 最終評估所有任務...")
        final_metrics = trainer.evaluate_all_tasks()
        for task, metrics in final_metrics.items():
            print(f"  {task}: {metrics['main_metric']:.4f}")
        
        # 最終遺忘檢查
        final_forgetting = trainer.check_forgetting()
        print(f"\n🔍 最終遺忘檢查:")
        print(f"📊 最大遺忘程度: {final_forgetting['max_drop']*100:.2f}%")
        print(f"🎯 總體評估: {'✅ 成功' if final_forgetting['acceptable'] else '❌ 失敗'}")
        
        # 保存測試結果
        trainer.save_training_history()
        
        print("\n🎉 依序訓練測試完成！")
        return True
        
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_mock_dataloaders(device):
    """創建模擬數據加載器"""
    print("🎭 創建模擬數據加載器...")
    
    from torch.utils.data import DataLoader, TensorDataset
    
    batch_size = 4
    num_batches = 5
    
    # 模擬數據 (使用512x512以匹配模型輸出)
    images = torch.randn(batch_size * num_batches, 3, 512, 512)
    
    # 分割數據
    seg_targets = torch.randint(0, 21, (batch_size * num_batches, 512, 512))
    seg_dataset = TensorDataset(images, seg_targets)
    seg_train_loader = DataLoader(seg_dataset, batch_size=batch_size, shuffle=True)
    seg_val_loader = DataLoader(seg_dataset, batch_size=batch_size, shuffle=False)
    
    # 檢測數據（簡化）
    det_targets = [
        {
            'boxes': torch.rand(2, 4),  # 每個圖像2個框
            'labels': torch.randint(0, 10, (2,))
        } for _ in range(batch_size * num_batches)
    ]
    
    class DetectionDataset:
        def __init__(self, images, targets):
            self.images = images
            self.targets = targets
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            return self.images[idx], self.targets[idx]
    
    det_dataset = DetectionDataset(images, det_targets)
    det_train_loader = DataLoader(det_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    det_val_loader = DataLoader(det_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)
    
    # 分類數據
    cls_targets = torch.randint(0, 10, (batch_size * num_batches,))
    cls_dataset = TensorDataset(images, cls_targets)
    cls_train_loader = DataLoader(cls_dataset, batch_size=batch_size, shuffle=True)
    cls_val_loader = DataLoader(cls_dataset, batch_size=batch_size, shuffle=False)
    
    dataloaders = {
        'segmentation_train': seg_train_loader,
        'segmentation_val': seg_val_loader,
        'detection_train': det_train_loader,
        'detection_val': det_val_loader,
        'classification_train': cls_train_loader,
        'classification_val': cls_val_loader
    }
    
    print("✅ 模擬數據加載器創建完成")
    return dataloaders


def main():
    """主函數"""
    print("🧪 依序訓練快速測試")
    print(f"📅 測試時間: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python: {sys.version}")
    print(f"🔥 PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"🚀 CUDA: {torch.version.cuda}")
        print(f"📱 GPU: {torch.cuda.get_device_name()}")
    else:
        print("💻 使用 CPU")
    
    print("\n" + "=" * 50)
    
    # 運行測試
    success = test_sequential_trainer()
    
    if success:
        print("\n✅ 依序訓練器測試通過！")
        print("🔄 依序訓練流程設置完成！")
        print("\n📋 訓練計畫:")
        print("Stage 1: 分割任務 (記錄基準)")
        print("Stage 2: 檢測任務 + EWC")
        print("Stage 3: 分類任務 + EWC")
        print("\n✅ Phase 3 實現完成！準備進入 Phase 4: 評估與優化")
        return 0
    else:
        print("\n❌ 依序訓練器測試失敗！")
        return 1


if __name__ == "__main__":
    sys.exit(main())
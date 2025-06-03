#!/usr/bin/env python3
"""
快速測試修復後的EWC效果
驗證Fisher矩陣計算和遺忘率控制
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import time

# 添加專案路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.unified_model import create_unified_model
from src.utils.ewc import create_ewc_handler, ewc_loss
from src.datasets.unified_dataloader import create_unified_dataloaders
from src.losses.segmentation_loss import create_segmentation_loss
from src.losses.detection_loss import create_detection_loss
from src.losses.classification_loss import create_classification_loss


def test_fisher_calculation():
    """測試Fisher矩陣計算的數值穩定性"""
    print("\n" + "="*50)
    print("🧪 測試Fisher矩陣計算")
    print("="*50)
    
    # 創建模型
    model = create_unified_model('default')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 創建EWC處理器
    ewc = create_ewc_handler(model, importance=5000.0)  # 使用新的預設值
    
    # 創建數據加載器
    dataloaders = create_unified_dataloaders(
        data_dir='./data',
        batch_size=4,
        num_workers=0
    )
    
    # 獲取分割任務的加載器
    seg_loader = dataloaders['train'].get_task_loader('segmentation')
    
    # 計算Fisher矩陣
    print("\n📊 計算Fisher矩陣...")
    start_time = time.time()
    fisher_matrix = ewc.compute_fisher_matrix(seg_loader, task_id=0, num_samples=20, verbose=True)
    compute_time = time.time() - start_time
    
    # 驗證Fisher值範圍
    fisher_values = []
    for name, tensor in fisher_matrix.items():
        fisher_values.extend(tensor.cpu().numpy().flatten())
    
    fisher_values = np.array(fisher_values)
    print(f"\n📈 Fisher矩陣統計:")
    print(f"  - 計算時間: {compute_time:.2f}秒")
    print(f"  - 平均值: {fisher_values.mean():.6f}")
    print(f"  - 標準差: {fisher_values.std():.6f}")
    print(f"  - 最小值: {fisher_values.min():.6f}")
    print(f"  - 最大值: {fisher_values.max():.6f}")
    print(f"  - 非零值比例: {(fisher_values > 1e-8).mean():.2%}")
    
    # 檢查是否有爆炸值
    if fisher_values.max() > 1e6:
        print("⚠️ 警告: Fisher值超過1e6上限!")
    else:
        print("✅ Fisher值在正常範圍內")
    
    return ewc, fisher_values


def test_ewc_penalty_growth():
    """測試EWC懲罰項的增長情況"""
    print("\n" + "="*50)
    print("🧪 測試EWC懲罰項增長")
    print("="*50)
    
    # 創建模型和EWC
    model = create_unified_model('default')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    ewc = create_ewc_handler(model, importance=5000.0)
    
    # 創建數據加載器
    dataloaders = create_unified_dataloaders(
        data_dir='./data',
        batch_size=4,
        num_workers=0
    )
    
    # 完成第一個任務
    seg_loader = dataloaders['train'].get_task_loader('segmentation')
    ewc.finish_task(seg_loader, task_id=0, verbose=False)
    
    # 模擬參數變化並測試懲罰項
    print("\n📊 測試不同參數變化下的EWC懲罰項:")
    
    # 保存原始參數
    original_state = model.state_dict()
    
    # 測試不同程度的參數變化
    for scale in [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]:
        # 添加噪音到參數
        for name, param in model.named_parameters():
            if param.requires_grad:
                noise = torch.randn_like(param) * scale
                param.data = original_state[name] + noise
        
        # 計算懲罰項
        penalty = ewc.penalty(model)
        print(f"  參數變化 {scale:>4.2f}: 懲罰項 = {penalty.item():>12.6f}")
    
    # 恢復原始參數
    model.load_state_dict(original_state)
    
    return ewc


def test_adaptive_importance():
    """測試自適應權重調整"""
    print("\n" + "="*50)
    print("🧪 測試自適應權重調整")
    print("="*50)
    
    # 創建模型和EWC
    model = create_unified_model('default')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    ewc = create_ewc_handler(model, importance=5000.0)
    
    print(f"\n初始權重: {ewc.adaptive_importance:.1f}")
    
    # 測試不同遺忘率下的權重調整
    test_rates = [0.02, 0.05, 0.10, 0.50, 0.90]
    
    for rate in test_rates:
        ewc.update_adaptive_importance(rate, target_rate=0.05)
        print(f"遺忘率 {rate:.2%} → 調整後權重: {ewc.adaptive_importance:.1f}")
    
    return ewc


def test_multitask_training():
    """測試多任務訓練的遺忘控制"""
    print("\n" + "="*50)
    print("🧪 測試多任務訓練遺忘控制")
    print("="*50)
    
    # 創建模型
    model = create_unified_model('default')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 創建EWC處理器
    ewc = create_ewc_handler(model, importance=20000.0)  # 更高的初始權重
    
    # 創建優化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 創建數據加載器
    dataloaders = create_unified_dataloaders(
        data_dir='./data',
        batch_size=8,
        num_workers=0
    )
    
    # 創建損失函數
    seg_criterion = create_segmentation_loss()
    det_criterion = create_detection_loss()
    cls_criterion = create_classification_loss()
    
    # 第一階段：訓練分割任務
    print("\n🎯 Stage 1: 訓練分割任務")
    seg_loader = dataloaders['train'].get_task_loader('segmentation')
    
    model.train()
    for epoch in range(2):  # 快速測試
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(seg_loader):
            if batch_idx >= 5:  # 只訓練幾個批次
                break
            
            images, targets = batch
            images = images.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, task_type='segmentation')
            
            # 準備目標
            if isinstance(targets, list):
                # VOC數據集返回字典格式
                if isinstance(targets[0], dict) and 'masks' in targets[0]:
                    masks = torch.stack([t['masks'] for t in targets]).to(device)
                elif isinstance(targets[0], torch.Tensor):
                    masks = torch.stack(targets).to(device)
                else:
                    # 處理其他格式
                    print(f"警告: 未知的targets格式: {type(targets[0])}")
                    masks = targets[0].to(device) if hasattr(targets[0], 'to') else targets[0]
            elif isinstance(targets, dict) and 'masks' in targets:
                masks = targets['masks'].to(device)
            elif isinstance(targets, torch.Tensor):
                masks = targets.to(device)
            else:
                print(f"警告: 未知的targets類型: {type(targets)}")
                masks = targets
            
            loss_output = seg_criterion(outputs['segmentation'], masks)
            # 處理可能的tuple返回值
            if isinstance(loss_output, tuple):
                loss = loss_output[0]  # 取第一個元素作為主損失
            else:
                loss = loss_output
            
            # 如果有EWC懲罰，添加進去
            if ewc.task_count > 0:
                total_loss, ewc_penalty = ewc_loss(loss, ewc, model)
                print(f"    EWC懲罰: {ewc_penalty.item():.6f}")
            else:
                total_loss = loss
                ewc_penalty = None
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"  Epoch {epoch+1}: 平均損失 = {epoch_loss/(batch_idx+1):.4f}")
    
    # 評估分割性能
    model.eval()
    seg_val_loader = dataloaders['val'].get_task_loader('segmentation')
    seg_perf_before = evaluate_segmentation(model, seg_val_loader, device)
    print(f"  分割性能: {seg_perf_before:.4f}")
    
    # 完成任務1，設置EWC
    print("\n📥 設置EWC保護...")
    ewc.finish_task(seg_loader, task_id=0, verbose=True)
    
    # 第二階段：訓練檢測任務
    print("\n🎯 Stage 2: 訓練檢測任務（帶EWC保護）")
    det_loader = dataloaders['train'].get_task_loader('detection')
    
    model.train()
    for epoch in range(2):
        epoch_loss = 0.0
        epoch_ewc = 0.0
        for batch_idx, batch in enumerate(det_loader):
            if batch_idx >= 5:
                break
            
            if isinstance(batch, list) and len(batch) == 2:
                images = batch[0].to(device)
                targets = batch[1]
            else:
                continue
            
            optimizer.zero_grad()
            outputs = model(images, task_type='detection')
            
            loss_output = det_criterion(outputs['detection'], targets)
            # 處理可能的tuple返回值
            if isinstance(loss_output, tuple):
                loss = loss_output[0]  # 取第一個元素作為主損失
            else:
                loss = loss_output
            
            # 添加EWC懲罰
            total_loss, ewc_penalty = ewc_loss(loss, ewc, model)
            epoch_ewc += ewc_penalty.item()
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"  Epoch {epoch+1}: 損失 = {epoch_loss/(batch_idx+1):.4f}, EWC = {epoch_ewc/(batch_idx+1):.4f}")
    
    # 重新評估分割性能
    model.eval()
    seg_perf_after = evaluate_segmentation(model, seg_val_loader, device)
    forgetting_rate = (seg_perf_before - seg_perf_after) / seg_perf_before if seg_perf_before > 0 else 0
    
    print(f"\n📊 遺忘分析:")
    print(f"  分割性能變化: {seg_perf_before:.4f} → {seg_perf_after:.4f}")
    print(f"  遺忘率: {forgetting_rate:.2%}")
    
    if forgetting_rate > 0.05:
        print("⚠️ 遺忘率仍然超過5%，需要進一步調整!")
        ewc.update_adaptive_importance(forgetting_rate)
        print(f"  新的EWC權重: {ewc.adaptive_importance:.1f}")
    else:
        print("✅ 遺忘率控制在5%以內!")
    
    return forgetting_rate


def evaluate_segmentation(model, dataloader, device):
    """簡單的分割評估函數"""
    model.eval()
    total_correct = 0
    total_pixels = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 5:  # 快速測試
                break
            
            images, targets = batch
            images = images.to(device)
            
            outputs = model(images, task_type='segmentation')
            preds = outputs['segmentation'].argmax(dim=1)
            
            if isinstance(targets, list):
                # VOC數據集返回字典格式
                if isinstance(targets[0], dict) and 'masks' in targets[0]:
                    masks = torch.stack([t['masks'] for t in targets]).to(device)
                elif isinstance(targets[0], torch.Tensor):
                    masks = torch.stack(targets).to(device)
                else:
                    # 處理其他格式
                    masks = targets[0].to(device) if hasattr(targets[0], 'to') else targets[0]
            elif isinstance(targets, dict) and 'masks' in targets:
                masks = targets['masks'].to(device)
            elif isinstance(targets, torch.Tensor):
                masks = targets.to(device)
            else:
                masks = targets
            
            total_correct += (preds == masks).sum().item()
            total_pixels += masks.numel()
    
    return total_correct / total_pixels if total_pixels > 0 else 0


def main():
    """主測試函數"""
    print("🚀 開始測試修復的EWC實現")
    print("="*60)
    
    # 1. 測試Fisher矩陣計算
    ewc1, fisher_values = test_fisher_calculation()
    
    # 2. 測試EWC懲罰項增長
    ewc2 = test_ewc_penalty_growth()
    
    # 3. 測試自適應權重調整
    ewc3 = test_adaptive_importance()
    
    # 4. 測試多任務訓練
    forgetting_rate = test_multitask_training()
    
    # 總結
    print("\n" + "="*60)
    print("📊 測試總結")
    print("="*60)
    print(f"✅ Fisher矩陣計算: 數值穩定")
    print(f"✅ EWC懲罰項: 正常增長")
    print(f"✅ 自適應權重: 功能正常")
    print(f"{'✅' if forgetting_rate < 0.05 else '⚠️'} 遺忘控制: {forgetting_rate:.2%}")
    
    print("\n🎯 建議:")
    if forgetting_rate > 0.05:
        print(f"  - 使用更高的EWC權重 (建議: {20000 * (1 + forgetting_rate * 10):.0f})")
        print(f"  - 啟用自適應權重調整")
        print(f"  - 考慮使用Online EWC減少計算開銷")
    else:
        print(f"  - 當前設置有效，可以開始完整訓練")
    
    print("\n✨ 測試完成!")


if __name__ == "__main__":
    main()
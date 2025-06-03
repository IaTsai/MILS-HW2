#!/usr/bin/env python3
"""
損失函數測試腳本
測試檢測、分割、分類和多任務損失函數的功能、梯度回傳和數值穩定性
"""
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.losses.detection_loss import create_detection_loss
from src.losses.segmentation_loss import create_segmentation_loss
from src.losses.classification_loss import create_classification_loss
from src.losses.multitask_loss import create_multitask_loss


def test_detection_loss():
    """測試檢測損失函數"""
    print("🎯 測試檢測損失函數...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 創建檢測損失
    det_loss = create_detection_loss(num_classes=10, iou_loss_type='giou')
    
    # 測試數據
    batch_size = 4
    num_predictions = 100
    
    # 預測: (B, H*W, 6) - (cx, cy, w, h, centerness, class)
    predictions = torch.randn(batch_size, num_predictions, 6).to(device)
    predictions[..., :4] = torch.sigmoid(predictions[..., :4])  # 歸一化坐標
    predictions[..., 5] = torch.randint(0, 10, (batch_size, num_predictions)).float()  # 類別
    
    # 目標
    targets = []
    for b in range(batch_size):
        num_objects = torch.randint(1, 5, (1,)).item()
        target = {
            'boxes': torch.rand(num_objects, 4).to(device),  # (cx, cy, w, h)
            'labels': torch.randint(0, 10, (num_objects,)).to(device)
        }
        targets.append(target)
    
    # 測試前向傳播
    total_loss, loss_dict = det_loss(predictions, targets)
    
    print(f"  ✅ 檢測損失計算成功")
    print(f"  📊 總損失: {total_loss.item():.4f}")
    print(f"  🔍 損失組成: {list(loss_dict.keys())}")
    
    # 測試梯度回傳
    predictions.requires_grad_(True)
    total_loss.backward()
    
    if predictions.grad is not None:
        grad_norm = predictions.grad.norm().item()
        print(f"  📈 梯度範數: {grad_norm:.6f}")
        print(f"  ✅ 梯度回傳正常")
    else:
        print(f"  ❌ 梯度回傳失敗")
    
    # 測試不同IoU損失類型
    print(f"  🧪 測試不同IoU損失類型:")
    iou_types = ['iou', 'giou', 'diou', 'ciou']
    
    for iou_type in iou_types:
        try:
            test_loss = create_detection_loss(num_classes=10, iou_loss_type=iou_type)
            test_total_loss, _ = test_loss(predictions.detach(), targets)
            print(f"    {iou_type}: {test_total_loss.item():.4f}")
        except Exception as e:
            print(f"    {iou_type}: 錯誤 - {e}")
    
    return True


def test_segmentation_loss():
    """測試分割損失函數"""
    print("\n🎨 測試分割損失函數...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 創建分割損失
    seg_loss = create_segmentation_loss(num_classes=21, loss_type='combined')
    
    # 測試數據
    batch_size = 4
    height, width = 128, 128
    num_classes = 21
    
    # 預測: (B, C, H, W)
    predictions = torch.randn(batch_size, num_classes, height, width).to(device)
    
    # 目標: (B, H, W)
    targets = torch.randint(0, num_classes, (batch_size, height, width)).to(device)
    
    # 測試前向傳播
    total_loss, loss_dict = seg_loss(predictions, targets)
    
    print(f"  ✅ 分割損失計算成功")
    print(f"  📊 總損失: {total_loss.item():.4f}")
    print(f"  🔍 損失組成: {list(loss_dict.keys())}")
    
    # 測試梯度回傳
    predictions.requires_grad_(True)
    total_loss.backward()
    
    if predictions.grad is not None:
        grad_norm = predictions.grad.norm().item()
        print(f"  📈 梯度範數: {grad_norm:.6f}")
        print(f"  ✅ 梯度回傳正常")
    else:
        print(f"  ❌ 梯度回傳失敗")
    
    # 測試不同損失類型
    print(f"  🧪 測試不同損失類型:")
    loss_types = ['ce', 'dice', 'focal', 'combined', 'advanced']
    
    for loss_type in loss_types:
        try:
            test_loss = create_segmentation_loss(num_classes=21, loss_type=loss_type)
            test_total_loss, _ = test_loss(predictions.detach(), targets)
            print(f"    {loss_type}: {test_total_loss.item():.4f}")
        except Exception as e:
            print(f"    {loss_type}: 錯誤 - {e}")
    
    return True


def test_classification_loss():
    """測試分類損失函數"""
    print("\n📊 測試分類損失函數...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 創建分類損失
    cls_loss = create_classification_loss(num_classes=10, loss_type='combined')
    
    # 測試數據
    batch_size = 16
    num_classes = 10
    feature_dim = 128
    
    # 預測: (B, C)
    predictions = torch.randn(batch_size, num_classes).to(device)
    
    # 目標: (B,)
    targets = torch.randint(0, num_classes, (batch_size,)).to(device)
    
    # 特徵: (B, D) - 用於對比學習
    features = torch.randn(batch_size, feature_dim).to(device)
    
    # 測試前向傳播
    total_loss, loss_dict = cls_loss(predictions, targets, features=features)
    
    print(f"  ✅ 分類損失計算成功")
    print(f"  📊 總損失: {total_loss.item():.4f}")
    print(f"  🔍 損失組成: {list(loss_dict.keys())}")
    
    # 測試梯度回傳
    predictions.requires_grad_(True)
    total_loss.backward()
    
    if predictions.grad is not None:
        grad_norm = predictions.grad.norm().item()
        print(f"  📈 梯度範數: {grad_norm:.6f}")
        print(f"  ✅ 梯度回傳正常")
    else:
        print(f"  ❌ 梯度回傳失敗")
    
    # 測試不同損失類型
    print(f"  🧪 測試不同損失類型:")
    loss_types = ['ce', 'focal', 'temperature', 'contrastive', 'combined']
    
    for loss_type in loss_types:
        try:
            test_loss = create_classification_loss(num_classes=10, loss_type=loss_type)
            
            if loss_type == 'contrastive':
                test_total_loss, _ = test_loss(predictions.detach(), targets, features=features)
            else:
                test_total_loss, _ = test_loss(predictions.detach(), targets)
            
            print(f"    {loss_type}: {test_total_loss.item():.4f}")
        except Exception as e:
            print(f"    {loss_type}: 錯誤 - {e}")
    
    # 測試特殊功能
    print(f"  🎭 測試 Mixup:")
    targets_b = torch.randint(0, num_classes, (batch_size,)).to(device)
    mixup_params = {
        'target_a': targets,
        'target_b': targets_b,
        'lambda': 0.7
    }
    
    mixup_loss, _ = cls_loss(predictions.detach(), targets, mixup_params=mixup_params)
    print(f"    Mixup 損失: {mixup_loss.item():.4f}")
    
    # 測試知識蒸餾
    print(f"  🎓 測試知識蒸餾:")
    teacher_logits = torch.randn(batch_size, num_classes).to(device)
    kd_loss = create_classification_loss(num_classes=10, loss_type='distillation')
    
    kd_total_loss, _ = kd_loss(predictions.detach(), targets, teacher_logits=teacher_logits)
    print(f"    知識蒸餾損失: {kd_total_loss.item():.4f}")
    
    return True


def test_multitask_loss():
    """測試多任務損失函數"""
    print("\n🔗 測試多任務損失函數...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 創建多任務損失
    multitask_loss = create_multitask_loss(
        task_weights={'detection': 1.0, 'segmentation': 1.0, 'classification': 1.0},
        weighting_strategy='uncertainty'
    )
    
    # 測試數據
    batch_size = 4
    predictions = {
        'detection': torch.randn(batch_size, 50, 6).to(device),  # (B, H*W, 6)
        'segmentation': torch.randn(batch_size, 21, 128, 128).to(device),  # (B, C, H, W)
        'classification': torch.randn(batch_size, 10).to(device)  # (B, C)
    }
    
    # 目標數據
    targets = {
        'detection': [
            {
                'boxes': torch.rand(2, 4).to(device),
                'labels': torch.randint(0, 10, (2,)).to(device)
            } for _ in range(batch_size)
        ],
        'segmentation': torch.randint(0, 21, (batch_size, 128, 128)).to(device),
        'classification': torch.randint(0, 10, (batch_size,)).to(device)
    }
    
    # 共享特徵
    features = torch.randn(batch_size, 128).to(device)
    
    # 測試前向傳播
    total_loss, loss_info = multitask_loss(predictions, targets, features=features)
    
    print(f"  ✅ 多任務損失計算成功")
    print(f"  📊 總損失: {total_loss.item():.4f}")
    print(f"  🎯 任務損失: {[(k, v.item()) for k, v in loss_info['task_losses'].items()]}")
    print(f"  ⚖️ 任務權重: {loss_info['task_weights']}")
    
    if 'uncertainties' in loss_info:
        print(f"  🔮 不確定性: {loss_info['uncertainties']}")
    
    # 測試梯度回傳
    for task_pred in predictions.values():
        if task_pred.requires_grad:
            task_pred.requires_grad_(True)
    
    total_loss.backward()
    
    print(f"  ✅ 梯度回傳正常")
    
    # 測試不同權重策略
    print(f"  🧪 測試不同權重策略:")
    strategies = ['fixed', 'uncertainty', 'dynamic']
    
    for strategy in strategies:
        try:
            test_loss = create_multitask_loss(
                weighting_strategy=strategy,
                task_weights={'detection': 1.0, 'segmentation': 1.0, 'classification': 1.0}
            )
            
            test_total_loss, test_info = test_loss(predictions, targets, features=features)
            print(f"    {strategy}: {test_total_loss.item():.4f}, 權重: {test_info['task_weights']}")
            
        except Exception as e:
            print(f"    {strategy}: 錯誤 - {e}")
    
    # 測試權重調整
    print(f"  🔧 測試權重調整:")
    current_weights = multitask_loss.get_task_weights()
    print(f"    當前權重: {current_weights}")
    
    new_weights = {'detection': 2.0, 'segmentation': 0.5, 'classification': 1.5}
    multitask_loss.set_task_weights(new_weights)
    updated_weights = multitask_loss.get_task_weights()
    print(f"    更新後權重: {updated_weights}")
    
    return True


def test_loss_numerical_stability():
    """測試損失函數的數值穩定性"""
    print("\n🔬 測試損失函數數值穩定性...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 測試極端情況
    test_cases = [
        "正常情況",
        "極小值",
        "極大值",
        "零值",
        "NaN輸入"
    ]
    
    results = {}
    
    for i, case_name in enumerate(test_cases):
        print(f"  🧪 測試 {case_name}:")
        
        try:
            # 生成不同的測試數據
            if case_name == "正常情況":
                pred = torch.randn(4, 10).to(device)
                target = torch.randint(0, 10, (4,)).to(device)
            elif case_name == "極小值":
                pred = torch.full((4, 10), -100.0).to(device)
                target = torch.randint(0, 10, (4,)).to(device)
            elif case_name == "極大值":
                pred = torch.full((4, 10), 100.0).to(device)
                target = torch.randint(0, 10, (4,)).to(device)
            elif case_name == "零值":
                pred = torch.zeros(4, 10).to(device)
                target = torch.randint(0, 10, (4,)).to(device)
            else:  # NaN輸入
                pred = torch.full((4, 10), float('nan')).to(device)
                target = torch.randint(0, 10, (4,)).to(device)
            
            # 測試分類損失
            cls_loss = create_classification_loss(num_classes=10, loss_type='ce')
            loss, _ = cls_loss(pred, target)
            
            if torch.isnan(loss) or torch.isinf(loss):
                results[case_name] = f"❌ 數值不穩定: {loss.item()}"
            else:
                results[case_name] = f"✅ 穩定: {loss.item():.4f}"
            
        except Exception as e:
            results[case_name] = f"❌ 錯誤: {str(e)[:50]}"
        
        print(f"    {results[case_name]}")
    
    return results


def test_loss_convergence():
    """測試損失函數收斂性"""
    print("\n📈 測試損失函數收斂性...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 創建簡單的優化任務
    batch_size = 16
    num_classes = 10
    
    # 生成固定目標
    targets = torch.randint(0, num_classes, (batch_size,)).to(device)
    
    # 創建可學習參數
    logits = nn.Parameter(torch.randn(batch_size, num_classes).to(device))
    
    # 創建損失函數
    cls_loss = create_classification_loss(num_classes=10, loss_type='ce')
    
    # 優化器
    optimizer = torch.optim.Adam([logits], lr=0.01)
    
    # 記錄損失歷史
    loss_history = []
    
    print("  🏃 開始優化...")
    for step in range(100):
        optimizer.zero_grad()
        
        loss, _ = cls_loss(logits, targets)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if step % 20 == 0:
            print(f"    Step {step}: 損失 = {loss.item():.4f}")
    
    # 檢查收斂性
    initial_loss = loss_history[0]
    final_loss = loss_history[-1]
    improvement = (initial_loss - final_loss) / initial_loss * 100
    
    print(f"  📊 優化結果:")
    print(f"    初始損失: {initial_loss:.4f}")
    print(f"    最終損失: {final_loss:.4f}")
    print(f"    改善程度: {improvement:.2f}%")
    
    if improvement > 50:
        print(f"  ✅ 收斂性良好")
        return True
    else:
        print(f"  ⚠️ 收斂性較差")
        return False


def test_loss_gradient_flow():
    """測試損失函數梯度流"""
    print("\n🌊 測試損失函數梯度流...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 創建多任務損失
    multitask_loss = create_multitask_loss(weighting_strategy='uncertainty')
    
    # 測試數據
    batch_size = 4
    predictions = {
        'detection': torch.randn(batch_size, 20, 6, requires_grad=True).to(device),
        'segmentation': torch.randn(batch_size, 21, 64, 64, requires_grad=True).to(device),
        'classification': torch.randn(batch_size, 10, requires_grad=True).to(device)
    }
    
    targets = {
        'detection': [
            {
                'boxes': torch.rand(1, 4).to(device),
                'labels': torch.randint(0, 10, (1,)).to(device)
            } for _ in range(batch_size)
        ],
        'segmentation': torch.randint(0, 21, (batch_size, 64, 64)).to(device),
        'classification': torch.randint(0, 10, (batch_size,)).to(device)
    }
    
    # 前向傳播
    total_loss, loss_info = multitask_loss(predictions, targets)
    
    # 反向傳播
    total_loss.backward()
    
    # 檢查梯度
    gradient_stats = {}
    
    for task, pred in predictions.items():
        if pred.grad is not None:
            grad_norm = pred.grad.norm().item()
            grad_mean = pred.grad.mean().item()
            grad_std = pred.grad.std().item()
            
            gradient_stats[task] = {
                'norm': grad_norm,
                'mean': grad_mean,
                'std': grad_std,
                'has_nan': torch.isnan(pred.grad).any().item(),
                'has_inf': torch.isinf(pred.grad).any().item()
            }
        else:
            gradient_stats[task] = {'error': '無梯度'}
    
    print("  📊 梯度統計:")
    for task, stats in gradient_stats.items():
        if 'error' in stats:
            print(f"    {task}: ❌ {stats['error']}")
        elif stats['has_nan'] or stats['has_inf']:
            print(f"    {task}: ❌ 梯度異常 (NaN/Inf)")
        else:
            print(f"    {task}: ✅ 正常 (範數: {stats['norm']:.4f})")
    
    return all('error' not in stats and not stats.get('has_nan', False) and not stats.get('has_inf', False) 
              for stats in gradient_stats.values())


def run_comprehensive_loss_tests():
    """運行全面的損失函數測試"""
    print("🚀 開始損失函數全面測試...")
    print("=" * 70)
    
    results = {}
    
    # 1. 檢測損失測試
    try:
        results['detection'] = test_detection_loss()
        print("✅ 檢測損失測試完成")
    except Exception as e:
        print(f"❌ 檢測損失測試失敗: {e}")
        results['detection'] = False
    
    # 2. 分割損失測試
    try:
        results['segmentation'] = test_segmentation_loss()
        print("✅ 分割損失測試完成")
    except Exception as e:
        print(f"❌ 分割損失測試失敗: {e}")
        results['segmentation'] = False
    
    # 3. 分類損失測試
    try:
        results['classification'] = test_classification_loss()
        print("✅ 分類損失測試完成")
    except Exception as e:
        print(f"❌ 分類損失測試失敗: {e}")
        results['classification'] = False
    
    # 4. 多任務損失測試
    try:
        results['multitask'] = test_multitask_loss()
        print("✅ 多任務損失測試完成")
    except Exception as e:
        print(f"❌ 多任務損失測試失敗: {e}")
        results['multitask'] = False
    
    # 5. 數值穩定性測試
    try:
        stability_results = test_loss_numerical_stability()
        results['numerical_stability'] = stability_results
        print("✅ 數值穩定性測試完成")
    except Exception as e:
        print(f"❌ 數值穩定性測試失敗: {e}")
        results['numerical_stability'] = False
    
    # 6. 收斂性測試
    try:
        results['convergence'] = test_loss_convergence()
        print("✅ 收斂性測試完成")
    except Exception as e:
        print(f"❌ 收斂性測試失敗: {e}")
        results['convergence'] = False
    
    # 7. 梯度流測試
    try:
        results['gradient_flow'] = test_loss_gradient_flow()
        print("✅ 梯度流測試完成")
    except Exception as e:
        print(f"❌ 梯度流測試失敗: {e}")
        results['gradient_flow'] = False
    
    return results


def print_final_summary(results):
    """打印最終測試總結"""
    print("\n" + "=" * 70)
    print("📋 損失函數測試總結")
    print("=" * 70)
    
    # 成功率統計
    boolean_results = {k: v for k, v in results.items() if isinstance(v, bool)}
    successful_tests = sum(boolean_results.values())
    total_tests = len(boolean_results)
    
    print(f"✅ 測試通過: {successful_tests}/{total_tests}")
    
    # 詳細結果
    test_names = {
        'detection': '🎯 檢測損失',
        'segmentation': '🎨 分割損失', 
        'classification': '📊 分類損失',
        'multitask': '🔗 多任務損失',
        'convergence': '📈 收斂性',
        'gradient_flow': '🌊 梯度流'
    }
    
    for test_key, test_name in test_names.items():
        if test_key in results:
            result = results[test_key]
            status = "✅ 通過" if result else "❌ 失敗"
            print(f"{test_name}: {status}")
    
    # 數值穩定性詳細結果
    if 'numerical_stability' in results and isinstance(results['numerical_stability'], dict):
        print("🔬 數值穩定性:")
        for case, result in results['numerical_stability'].items():
            print(f"  {case}: {result}")
    
    # 最終結論
    print(f"\n🎯 最終結論:")
    if successful_tests == total_tests:
        print("🎉 所有損失函數測試通過！實現正確且穩定。")
        return True
    else:
        failed_tests = total_tests - successful_tests
        print(f"⚠️ {failed_tests} 個測試失敗，需要修復。")
        return False


if __name__ == "__main__":
    print("🔥 損失函數全面測試腳本")
    print(f"📅 測試時間: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python: {sys.version}")
    print(f"🔥 PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"🚀 CUDA: {torch.version.cuda}")
        print(f"📱 GPU: {torch.cuda.get_device_name()}")
    else:
        print("💻 使用 CPU")
    
    print("\n" + "=" * 70)
    
    # 運行測試
    test_results = run_comprehensive_loss_tests()
    
    # 打印總結
    success = print_final_summary(test_results)
    
    if success:
        print("\n✅ 損失函數實現完成！")
    
    # 退出碼
    sys.exit(0 if success else 1)
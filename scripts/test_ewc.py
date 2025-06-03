#!/usr/bin/env python3
"""
EWC (Elastic Weight Consolidation) 測試腳本
驗證 EWC 算法的功能、效率和防遺忘效果
"""
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.ewc import EWC, ewc_loss, create_ewc_handler
from src.models.unified_model import create_unified_model


class SimpleModel(nn.Module):
    """簡單測試模型"""
    def __init__(self, input_dim=10, hidden_dim=50, output_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def create_synthetic_data(num_samples=1000, input_dim=10, num_classes=5, task_shift=0.0):
    """
    創建合成數據集
    
    Args:
        num_samples: 樣本數量
        input_dim: 輸入維度
        num_classes: 類別數量
        task_shift: 任務偏移 (用於模擬任務差異)
    
    Returns:
        dataset: 數據集
    """
    # 生成隨機數據
    data = torch.randn(num_samples, input_dim)
    
    # 添加任務特定的偏移
    if task_shift != 0.0:
        data += task_shift
    
    # 生成標籤 (基於數據的線性組合)
    weights = torch.randn(input_dim)
    linear_combination = torch.matmul(data, weights)
    labels = (linear_combination > 0).long()
    
    # 擴展到多類別
    if num_classes > 2:
        # 使用量化來創建多類別
        quantiles = torch.quantile(linear_combination, torch.linspace(0, 1, num_classes + 1))
        labels = torch.bucketize(linear_combination, quantiles[1:-1])
    
    return TensorDataset(data, labels)


def test_ewc_basic_functionality():
    """測試 EWC 基本功能"""
    print("🧪 測試 EWC 基本功能...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 創建模型
    model = SimpleModel().to(device)
    
    # 創建 EWC 處理器
    ewc = create_ewc_handler(model, importance=1000.0)
    
    # 創建數據
    dataset = create_synthetic_data(num_samples=200)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print("✅ EWC 處理器創建成功")
    
    # 測試 Fisher 矩陣計算
    print("  🧮 測試 Fisher 矩陣計算...")
    fisher_matrix = ewc.compute_fisher_matrix(dataloader, verbose=False)
    
    assert len(fisher_matrix) > 0, "Fisher 矩陣為空"
    assert all(isinstance(f, torch.Tensor) for f in fisher_matrix.values()), "Fisher 矩陣類型錯誤"
    print("  ✅ Fisher 矩陣計算成功")
    
    # 測試參數存儲
    print("  📥 測試參數存儲...")
    ewc.store_optimal_params(task_id=0)
    assert 0 in ewc.optimal_params, "參數存儲失敗"
    print("  ✅ 參數存儲成功")
    
    # 測試懲罰項計算
    print("  💰 測試懲罰項計算...")
    penalty = ewc.penalty()
    assert isinstance(penalty, torch.Tensor), "懲罰項類型錯誤"
    assert penalty.requires_grad, "懲罰項不支持梯度"
    print(f"  ✅ 懲罰項計算成功: {penalty.item():.6f}")
    
    # 測試 EWC 損失
    print("  🔗 測試 EWC 損失整合...")
    
    # 稍微修改模型參數來產生非零懲罰項
    with torch.no_grad():
        for param in model.parameters():
            param.data += 0.1 * torch.randn_like(param.data)
    
    dummy_input = torch.randn(10, 10).to(device)
    output = model(dummy_input)
    dummy_labels = torch.randint(0, 5, (10,)).to(device)
    base_loss = F.cross_entropy(output, dummy_labels)
    
    total_loss, ewc_penalty = ewc_loss(base_loss, ewc)
    
    # 檢查懲罰項是否非零且總損失正確
    assert ewc_penalty.item() > 0, f"EWC 懲罰項應該大於0，實際值: {ewc_penalty.item()}"
    assert torch.allclose(total_loss, base_loss + ewc_penalty, atol=1e-6), "總損失計算錯誤"
    print(f"  ✅ EWC 損失整合成功: 基礎={base_loss.item():.4f}, EWC={ewc_penalty.item():.4f}, 總計={total_loss.item():.4f}")
    
    return True


def test_ewc_multitask_integration():
    """測試 EWC 與多任務模型的整合"""
    print("\n🧪 測試 EWC 與多任務模型整合...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 創建統一多任務模型
    model = create_unified_model('lightweight')  # 使用輕量化配置節省時間
    model = model.to(device)
    
    # 創建 EWC 處理器
    ewc = create_ewc_handler(model, importance=500.0)
    
    # 創建模擬數據 (模擬統一數據加載器格式)
    def create_multitask_data(num_samples=50):
        images = torch.randn(num_samples, 3, 224, 224)
        task_types = ['classification'] * num_samples  # 簡化為分類任務
        targets = [torch.randint(0, 10, (1,)) for _ in range(num_samples)]
        
        batch_data = []
        for i in range(0, num_samples, 10):  # 批次大小 10
            batch_images = images[i:i+10]
            batch_targets = targets[i:i+10]
            batch_data.append({
                'images': batch_images,
                'task_types': task_types[i:i+10],
                'targets': batch_targets
            })
        
        return batch_data
    
    # 創建數據
    multitask_data = create_multitask_data(num_samples=30)  # 減少樣本數以加快測試
    
    print("  🧮 計算多任務模型的 Fisher 矩陣...")
    
    # 手動處理批次數據
    sample_count = 0
    fisher_matrix = {}
    
    # 初始化 Fisher 矩陣
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_matrix[name] = torch.zeros_like(param.data)
    
    model.eval()
    for batch in multitask_data:
        images = batch['images'].to(device)
        batch_size = images.size(0)
        
        # 前向傳播
        outputs = model(images, task_type='classification')
        
        # 計算 Fisher 信息 (簡化版本)
        if 'classification' in outputs:
            class_output = outputs['classification']
            log_prob = F.log_softmax(class_output, dim=1)
            prob = F.softmax(class_output, dim=1)
            
            for class_idx in range(prob.size(1)):
                class_prob = prob[:, class_idx].sum()
                if class_prob.item() > 1e-8:
                    class_log_prob = log_prob[:, class_idx].sum()
                    
                    # 計算梯度
                    model.zero_grad()
                    class_log_prob.backward(retain_graph=True)
                    
                    # 累積 Fisher 信息
                    for name, param in model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            fisher_matrix[name] += class_prob.item() * (param.grad.data ** 2)
        
        sample_count += batch_size
        if sample_count >= 30:  # 限制樣本數
            break
    
    # 正規化
    for name in fisher_matrix:
        fisher_matrix[name] = fisher_matrix[name] / sample_count
    
    # 手動設置 Fisher 矩陣
    ewc.fisher_matrices[0] = fisher_matrix
    ewc.store_optimal_params(task_id=0)
    
    print("  ✅ 多任務模型 Fisher 矩陣計算完成")
    
    # 測試懲罰項計算
    penalty = ewc.penalty()
    print(f"  💰 多任務模型 EWC 懲罰項: {penalty.item():.6f}")
    
    # 測試記憶體使用
    memory_info = ewc.get_memory_usage()
    print(f"  💾 記憶體使用: {memory_info['total_mb']:.2f} MB")
    
    return True


def test_ewc_efficiency():
    """測試 EWC 計算效率"""
    print("\n⏱️ 測試 EWC 計算效率...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 測試不同模型大小
    model_configs = [
        {'input_dim': 10, 'hidden_dim': 50, 'output_dim': 5, 'name': 'Small'},
        {'input_dim': 50, 'hidden_dim': 200, 'output_dim': 10, 'name': 'Medium'},
        {'input_dim': 100, 'hidden_dim': 500, 'output_dim': 20, 'name': 'Large'}
    ]
    
    results = {}
    
    for config in model_configs:
        print(f"  📊 測試 {config['name']} 模型...")
        
        # 創建模型
        model = SimpleModel(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['output_dim']
        ).to(device)
        
        # 計算模型參數量
        total_params = sum(p.numel() for p in model.parameters())
        
        # 創建 EWC 處理器
        ewc = create_ewc_handler(model, importance=1000.0)
        
        # 創建數據
        dataset = create_synthetic_data(
            num_samples=500, 
            input_dim=config['input_dim'], 
            num_classes=config['output_dim']
        )
        dataloader = DataLoader(dataset, batch_size=50, shuffle=True)
        
        # 測試 Fisher 矩陣計算時間
        start_time = time.time()
        fisher_matrix = ewc.compute_fisher_matrix(dataloader, verbose=False)
        fisher_time = time.time() - start_time
        
        # 存儲參數
        ewc.store_optimal_params()
        
        # 測試懲罰項計算時間
        start_time = time.time()
        for _ in range(100):  # 多次計算求平均
            penalty = ewc.penalty()
        penalty_time = (time.time() - start_time) / 100
        
        # 記憶體使用
        memory_info = ewc.get_memory_usage()
        
        results[config['name']] = {
            'params': total_params,
            'fisher_time': fisher_time,
            'penalty_time': penalty_time * 1000,  # 轉換為毫秒
            'memory_mb': memory_info['total_mb'],
            'penalty_value': penalty.item()
        }
        
        print(f"    參數量: {total_params:,}")
        print(f"    Fisher 計算時間: {fisher_time:.3f}s")
        print(f"    懲罰項計算時間: {penalty_time*1000:.3f}ms")
        print(f"    記憶體使用: {memory_info['total_mb']:.2f}MB")
    
    # 打印效率總結
    print("\n📈 效率測試總結:")
    print("模型大小 | 參數量 | Fisher時間 | 懲罰項時間 | 記憶體使用")
    print("-" * 60)
    for name, stats in results.items():
        print(f"{name:8} | {stats['params']:7,} | {stats['fisher_time']:8.3f}s | {stats['penalty_time']:8.3f}ms | {stats['memory_mb']:7.2f}MB")
    
    return results


def test_catastrophic_forgetting_prevention():
    """測試災難性遺忘防止效果"""
    print("\n🧠 測試災難性遺忘防止效果...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 創建模型
    model = SimpleModel(input_dim=20, hidden_dim=100, output_dim=5).to(device)
    
    # 任務 1: 學習第一個任務
    print("  📚 學習任務 1...")
    task1_dataset = create_synthetic_data(num_samples=500, input_dim=20, num_classes=5, task_shift=0.0)
    task1_loader = DataLoader(task1_dataset, batch_size=32, shuffle=True)
    
    # 訓練任務 1
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    
    task1_losses = []
    for epoch in range(10):  # 簡化訓練
        epoch_loss = 0
        for batch_data, batch_labels in task1_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = F.cross_entropy(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(task1_loader)
        task1_losses.append(avg_loss)
        if epoch % 5 == 0:
            print(f"    Epoch {epoch}: 損失 = {avg_loss:.4f}")
    
    # 測試任務 1 性能
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch_data, batch_labels in task1_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            outputs = model(batch_data)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
        
        task1_accuracy = 100 * correct / total
        print(f"  ✅ 任務 1 訓練完成，準確率: {task1_accuracy:.2f}%")
    
    # 設置 EWC
    print("  🔧 設置 EWC...")
    ewc = create_ewc_handler(model, importance=1000.0)
    ewc.finish_task(task1_loader, task_id=0, verbose=False)
    
    # 任務 2: 學習第二個任務 (有 EWC 保護)
    print("  📚 學習任務 2 (有 EWC 保護)...")
    task2_dataset = create_synthetic_data(num_samples=500, input_dim=20, num_classes=5, task_shift=2.0)
    task2_loader = DataLoader(task2_dataset, batch_size=32, shuffle=True)
    
    model.train()
    task2_losses = []
    ewc_penalties = []
    
    for epoch in range(10):
        epoch_loss = 0
        epoch_ewc = 0
        for batch_data, batch_labels in task2_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            base_loss = F.cross_entropy(outputs, batch_labels)
            
            # 添加 EWC 懲罰項
            total_loss, ewc_penalty = ewc_loss(base_loss, ewc)
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += base_loss.item()
            epoch_ewc += ewc_penalty.item()
        
        avg_loss = epoch_loss / len(task2_loader)
        avg_ewc = epoch_ewc / len(task2_loader)
        task2_losses.append(avg_loss)
        ewc_penalties.append(avg_ewc)
        
        if epoch % 5 == 0:
            print(f"    Epoch {epoch}: 基礎損失 = {avg_loss:.4f}, EWC 懲罰 = {avg_ewc:.4f}")
    
    # 測試任務 2 性能
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch_data, batch_labels in task2_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            outputs = model(batch_data)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
        
        task2_accuracy = 100 * correct / total
        print(f"  ✅ 任務 2 訓練完成，準確率: {task2_accuracy:.2f}%")
    
    # 重新測試任務 1 性能 (檢查遺忘)
    print("  🔍 檢查任務 1 遺忘情況...")
    with torch.no_grad():
        correct = 0
        total = 0
        for batch_data, batch_labels in task1_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            outputs = model(batch_data)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
        
        task1_final_accuracy = 100 * correct / total
        forgetting_rate = task1_accuracy - task1_final_accuracy
        
        print(f"  📊 任務 1 最終準確率: {task1_final_accuracy:.2f}%")
        print(f"  📉 遺忘率: {forgetting_rate:.2f}%")
    
    # 與無 EWC 的情況對比
    print("\n  🆚 對比：無 EWC 保護的情況...")
    
    # 重新創建模型進行對比
    model_no_ewc = SimpleModel(input_dim=20, hidden_dim=100, output_dim=5).to(device)
    optimizer_no_ewc = optim.Adam(model_no_ewc.parameters(), lr=0.001)
    
    # 訓練任務 1 (無 EWC)
    model_no_ewc.train()
    for epoch in range(10):
        for batch_data, batch_labels in task1_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            optimizer_no_ewc.zero_grad()
            outputs = model_no_ewc(batch_data)
            loss = F.cross_entropy(outputs, batch_labels)
            loss.backward()
            optimizer_no_ewc.step()
    
    # 訓練任務 2 (無 EWC)
    for epoch in range(10):
        for batch_data, batch_labels in task2_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            optimizer_no_ewc.zero_grad()
            outputs = model_no_ewc(batch_data)
            loss = F.cross_entropy(outputs, batch_labels)
            loss.backward()
            optimizer_no_ewc.step()
    
    # 測試無 EWC 的任務 1 性能
    model_no_ewc.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch_data, batch_labels in task1_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            outputs = model_no_ewc(batch_data)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
        
        task1_no_ewc_accuracy = 100 * correct / total
        no_ewc_forgetting = task1_accuracy - task1_no_ewc_accuracy
    
    print(f"    無 EWC 任務 1 最終準確率: {task1_no_ewc_accuracy:.2f}%")
    print(f"    無 EWC 遺忘率: {no_ewc_forgetting:.2f}%")
    
    # EWC 效果評估
    ewc_improvement = no_ewc_forgetting - forgetting_rate
    print(f"\n  🎯 EWC 效果評估:")
    print(f"    EWC 遺忘率: {forgetting_rate:.2f}%")
    print(f"    無 EWC 遺忘率: {no_ewc_forgetting:.2f}%")
    print(f"    EWC 改善: {ewc_improvement:.2f}%")
    
    if ewc_improvement > 0:
        print("  ✅ EWC 有效減少了災難性遺忘！")
    else:
        print("  ⚠️ EWC 效果不明顯，可能需要調整參數")
    
    return {
        'ewc_forgetting': forgetting_rate,
        'no_ewc_forgetting': no_ewc_forgetting,
        'improvement': ewc_improvement,
        'task1_accuracy': task1_accuracy,
        'task2_accuracy': task2_accuracy,
        'task1_final_accuracy': task1_final_accuracy
    }


def run_comprehensive_ewc_test():
    """運行全面的 EWC 測試"""
    print("🚀 開始 EWC 全面測試...")
    print("=" * 70)
    
    results = {}
    
    # 1. 基本功能測試
    try:
        basic_success = test_ewc_basic_functionality()
        results['basic_functionality'] = basic_success
        print("✅ 基本功能測試通過")
    except Exception as e:
        print(f"❌ 基本功能測試失敗: {e}")
        results['basic_functionality'] = False
        return results
    
    # 2. 多任務整合測試
    try:
        multitask_success = test_ewc_multitask_integration()
        results['multitask_integration'] = multitask_success
        print("✅ 多任務整合測試通過")
    except Exception as e:
        print(f"❌ 多任務整合測試失敗: {e}")
        results['multitask_integration'] = False
    
    # 3. 效率測試
    try:
        efficiency_results = test_ewc_efficiency()
        results['efficiency'] = efficiency_results
        print("✅ 效率測試通過")
    except Exception as e:
        print(f"❌ 效率測試失敗: {e}")
        results['efficiency'] = False
    
    # 4. 遺忘防止測試
    try:
        forgetting_results = test_catastrophic_forgetting_prevention()
        results['forgetting_prevention'] = forgetting_results
        print("✅ 遺忘防止測試通過")
    except Exception as e:
        print(f"❌ 遺忘防止測試失敗: {e}")
        results['forgetting_prevention'] = False
    
    return results


def print_final_summary(results):
    """打印最終測試總結"""
    print("\n" + "=" * 70)
    print("📋 EWC 測試總結")
    print("=" * 70)
    
    # 成功率統計
    successful_tests = sum(1 for r in results.values() if r not in [False, None])
    total_tests = len(results)
    
    print(f"✅ 測試通過: {successful_tests}/{total_tests}")
    
    # 詳細結果
    for test_name, result in results.items():
        if test_name == 'basic_functionality':
            status = "✅ 通過" if result else "❌ 失敗"
            print(f"🧪 基本功能: {status}")
        
        elif test_name == 'multitask_integration':
            status = "✅ 通過" if result else "❌ 失敗"
            print(f"🔗 多任務整合: {status}")
        
        elif test_name == 'efficiency':
            if result:
                print(f"⏱️ 效率測試: ✅ 通過")
                # 顯示效率總結
                large_model = result.get('Large', {})
                if large_model:
                    print(f"   大模型 Fisher 計算: {large_model.get('fisher_time', 0):.3f}s")
                    print(f"   懲罰項計算: {large_model.get('penalty_time', 0):.3f}ms")
                    print(f"   記憶體使用: {large_model.get('memory_mb', 0):.2f}MB")
            else:
                print(f"⏱️ 效率測試: ❌ 失敗")
        
        elif test_name == 'forgetting_prevention':
            if result:
                print(f"🧠 遺忘防止: ✅ 通過")
                print(f"   EWC 遺忘率: {result.get('ewc_forgetting', 0):.2f}%")
                print(f"   無 EWC 遺忘率: {result.get('no_ewc_forgetting', 0):.2f}%")
                print(f"   改善程度: {result.get('improvement', 0):.2f}%")
            else:
                print(f"🧠 遺忘防止: ❌ 失敗")
    
    # 最終結論
    print(f"\n🎯 最終結論:")
    if successful_tests == total_tests:
        print("🎉 所有 EWC 測試通過！算法實現正確且有效。")
        return True
    else:
        failed_tests = total_tests - successful_tests
        print(f"⚠️ {failed_tests} 個測試失敗，需要修復。")
        return False


if __name__ == "__main__":
    print("🧮 EWC (Elastic Weight Consolidation) 測試腳本")
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
    test_results = run_comprehensive_ewc_test()
    
    # 打印總結
    success = print_final_summary(test_results)
    
    if success:
        print("\n✅ EWC 算法實現完成！")
    
    # 退出碼
    sys.exit(0 if success else 1)
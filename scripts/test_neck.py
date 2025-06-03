#!/usr/bin/env python3
"""
頸部網路測試腳本
測試 FPN 頸部網路的功能、性能和參數量
"""
import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.backbone import create_backbone
from src.models.neck import create_neck, FeaturePyramidNetwork, LightweightNeck


def test_neck_basic_functionality():
    """測試頸部網路基本功能"""
    print("🧪 測試頸部網路基本功能...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 使用設備: {device}")
    
    # 創建 FPN 頸部網路
    fpn = create_neck('fpn', in_channels_list=[16, 24, 48, 96], out_channels=256)
    fpn = fpn.to(device)
    
    # 模擬骨幹網路輸出
    batch_size = 2
    input_features = {
        'layer1': torch.randn(batch_size, 16, 128, 128).to(device),  # 1/4
        'layer2': torch.randn(batch_size, 24, 64, 64).to(device),    # 1/8
        'layer3': torch.randn(batch_size, 48, 32, 32).to(device),    # 1/16
        'layer4': torch.randn(batch_size, 96, 16, 16).to(device)     # 1/32
    }
    
    # 前向傳播
    with torch.no_grad():
        output_features = fpn(input_features)
    
    # 驗證輸出
    expected_shapes = {
        'P2': (batch_size, 256, 128, 128),
        'P3': (batch_size, 256, 64, 64),
        'P4': (batch_size, 256, 32, 32),
        'P5': (batch_size, 256, 16, 16)
    }
    
    print("✅ FPN 基本功能測試:")
    print(f"  📊 參數統計: {fpn.get_parameter_count()}")
    print(f"  📋 特徵信息: {fpn.get_feature_info()}")
    
    success = True
    for name, expected_shape in expected_shapes.items():
        if name in output_features:
            actual_shape = tuple(output_features[name].shape)
            if actual_shape == expected_shape:
                print(f"  ✅ {name}: {actual_shape}")
            else:
                print(f"  ❌ {name}: 期望 {expected_shape}, 實際 {actual_shape}")
                success = False
        else:
            print(f"  ❌ 缺少輸出特徵: {name}")
            success = False
    
    return success, fpn


def test_lightweight_neck():
    """測試輕量化頸部網路"""
    print("\n🧪 測試輕量化頸部網路...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 創建輕量化頸部網路
    lightweight_neck = create_neck('lightweight', in_channels_list=[16, 24, 48, 96], out_channels=128)
    lightweight_neck = lightweight_neck.to(device)
    
    # 模擬輸入
    batch_size = 4
    input_features = {
        'layer1': torch.randn(batch_size, 16, 128, 128).to(device),
        'layer2': torch.randn(batch_size, 24, 64, 64).to(device),
        'layer3': torch.randn(batch_size, 48, 32, 32).to(device),
        'layer4': torch.randn(batch_size, 96, 16, 16).to(device)
    }
    
    # 前向傳播
    with torch.no_grad():
        output_features = lightweight_neck(input_features)
    
    print("✅ 輕量化頸部網路測試:")
    print(f"  📊 參數統計: {lightweight_neck.get_parameter_count()}")
    
    for name, feature in output_features.items():
        print(f"  📏 {name}: {feature.shape}")
    
    return lightweight_neck


def test_backbone_neck_integration():
    """測試骨幹網路與頸部網路整合"""
    print("\n🧪 測試骨幹網路與頸部網路整合...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 創建骨幹網路
    backbone = create_backbone('mobilenetv3_small', pretrained=False)
    backbone = backbone.to(device)
    
    # 創建頸部網路
    fpn = create_neck('fpn', in_channels_list=[16, 24, 48, 96], out_channels=256)
    fpn = fpn.to(device)
    
    # 測試輸入
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 512, 512).to(device)
    
    # 完整前向傳播
    with torch.no_grad():
        # 骨幹網路特徵提取
        backbone_features = backbone(input_tensor)
        
        # 頸部網路特徵融合
        neck_features = fpn(backbone_features)
    
    print("✅ 骨幹網路與頸部網路整合測試:")
    print("  🔗 骨幹網路輸出:")
    for name, feature in backbone_features.items():
        print(f"    {name}: {feature.shape}")
    
    print("  🔗 頸部網路輸出:")
    for name, feature in neck_features.items():
        print(f"    {name}: {feature.shape}")
    
    return backbone, fpn


def test_gradient_flow():
    """測試梯度流動"""
    print("\n🧪 測試梯度流動...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 創建網路
    fpn = create_neck('fpn', in_channels_list=[16, 24, 48, 96], out_channels=256)
    fpn = fpn.to(device)
    fpn.train()
    
    # 模擬輸入（需要梯度）
    input_features = {
        'layer1': torch.randn(1, 16, 128, 128, device=device, requires_grad=True),
        'layer2': torch.randn(1, 24, 64, 64, device=device, requires_grad=True),
        'layer3': torch.randn(1, 48, 32, 32, device=device, requires_grad=True),
        'layer4': torch.randn(1, 96, 16, 16, device=device, requires_grad=True)
    }
    
    # 前向傳播
    output_features = fpn(input_features)
    
    # 計算損失（簡單的均值）
    loss = sum(feature.mean() for feature in output_features.values())
    
    # 反向傳播
    loss.backward()
    
    print("✅ 梯度流動測試:")
    print(f"  📉 總損失: {loss.item():.6f}")
    
    # 檢查梯度
    gradient_stats = {}
    for name, param in fpn.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            gradient_stats[name] = grad_norm
            if grad_norm > 0:
                print(f"  ✅ {name}: 梯度範數 {grad_norm:.6f}")
            else:
                print(f"  ⚠️ {name}: 梯度為零")
        else:
            print(f"  ❌ {name}: 無梯度")
    
    return len([g for g in gradient_stats.values() if g > 0]) > 0


def test_parameter_efficiency():
    """測試參數效率"""
    print("\n🧪 測試參數效率...")
    
    # 比較不同配置的參數量
    configs = [
        {'name': 'FPN-256', 'type': 'fpn', 'out_channels': 256},
        {'name': 'FPN-128', 'type': 'fpn', 'out_channels': 128},
        {'name': 'FPN-64', 'type': 'fpn', 'out_channels': 64},
        {'name': 'Lightweight-128', 'type': 'lightweight', 'out_channels': 128},
        {'name': 'Lightweight-64', 'type': 'lightweight', 'out_channels': 64},
    ]
    
    print("✅ 參數效率比較:")
    
    results = []
    for config in configs:
        neck = create_neck(
            config['type'], 
            in_channels_list=[16, 24, 48, 96], 
            out_channels=config['out_channels']
        )
        
        param_count = neck.get_parameter_count()
        total_params = param_count['total_parameters']
        
        print(f"  📊 {config['name']}: {total_params:,} 參數 ({total_params/1e6:.3f}M)")
        
        results.append({
            'name': config['name'],
            'type': config['type'],
            'out_channels': config['out_channels'],
            'parameters': total_params
        })
    
    # 找出最佳平衡點
    print("\n🎯 推薦配置:")
    for result in results:
        if result['parameters'] < 1e6:  # < 1M 參數
            efficiency = result['out_channels'] / (result['parameters'] / 1e6)
            print(f"  🏆 {result['name']}: {efficiency:.1f} 通道/M參數")
    
    return results


def test_inference_speed():
    """測試推理速度"""
    print("\n🧪 測試推理速度...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 創建網路
    fpn = create_neck('fpn', in_channels_list=[16, 24, 48, 96], out_channels=256)
    fpn = fpn.to(device)
    fpn.eval()
    
    # 準備輸入
    input_features = {
        'layer1': torch.randn(1, 16, 128, 128).to(device),
        'layer2': torch.randn(1, 24, 64, 64).to(device),
        'layer3': torch.randn(1, 48, 32, 32).to(device),
        'layer4': torch.randn(1, 96, 16, 16).to(device)
    }
    
    # 熱身
    with torch.no_grad():
        for _ in range(10):
            _ = fpn(input_features)
    
    # 計時
    times = []
    with torch.no_grad():
        for _ in range(100):
            start_time = time.time()
            _ = fpn(input_features)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # 轉換為毫秒
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print("✅ 推理速度測試:")
    print(f"  ⏱️ 平均推理時間: {avg_time:.3f} ± {std_time:.3f} ms")
    print(f"  🚀 FPS: {1000/avg_time:.1f}")
    
    return avg_time < 50  # 目標 < 50ms


def run_comprehensive_tests():
    """運行全面測試"""
    print("🚀 開始頸部網路全面測試...")
    print("=" * 60)
    
    test_results = {}
    
    # 1. 基本功能測試
    try:
        success, fpn_model = test_neck_basic_functionality()
        test_results['basic_functionality'] = success
    except Exception as e:
        print(f"❌ 基本功能測試失敗: {e}")
        test_results['basic_functionality'] = False
        return test_results
    
    # 2. 輕量化測試
    try:
        lightweight_model = test_lightweight_neck()
        test_results['lightweight'] = True
    except Exception as e:
        print(f"❌ 輕量化網路測試失敗: {e}")
        test_results['lightweight'] = False
    
    # 3. 整合測試
    try:
        backbone_model, neck_model = test_backbone_neck_integration()
        test_results['integration'] = True
    except Exception as e:
        print(f"❌ 整合測試失敗: {e}")
        test_results['integration'] = False
    
    # 4. 梯度流動測試
    try:
        gradient_success = test_gradient_flow()
        test_results['gradient_flow'] = gradient_success
    except Exception as e:
        print(f"❌ 梯度流動測試失敗: {e}")
        test_results['gradient_flow'] = False
    
    # 5. 參數效率測試
    try:
        param_results = test_parameter_efficiency()
        test_results['parameter_efficiency'] = True
        test_results['param_details'] = param_results
    except Exception as e:
        print(f"❌ 參數效率測試失敗: {e}")
        test_results['parameter_efficiency'] = False
    
    # 6. 推理速度測試
    try:
        speed_success = test_inference_speed()
        test_results['inference_speed'] = speed_success
    except Exception as e:
        print(f"❌ 推理速度測試失敗: {e}")
        test_results['inference_speed'] = False
    
    return test_results


def print_final_summary(test_results):
    """打印最終總結"""
    print("\n" + "=" * 60)
    print("📋 頸部網路測試總結")
    print("=" * 60)
    
    total_tests = len([k for k in test_results.keys() if k != 'param_details'])
    passed_tests = sum(1 for k, v in test_results.items() if k != 'param_details' and v)
    
    print(f"✅ 測試通過: {passed_tests}/{total_tests}")
    
    if test_results.get('basic_functionality', False):
        print("✅ 基本功能: PASS")
    else:
        print("❌ 基本功能: FAIL")
    
    if test_results.get('lightweight', False):
        print("✅ 輕量化網路: PASS")
    else:
        print("❌ 輕量化網路: FAIL")
    
    if test_results.get('integration', False):
        print("✅ 骨幹網路整合: PASS")
    else:
        print("❌ 骨幹網路整合: FAIL")
    
    if test_results.get('gradient_flow', False):
        print("✅ 梯度流動: PASS")
    else:
        print("❌ 梯度流動: FAIL")
    
    if test_results.get('parameter_efficiency', False):
        print("✅ 參數效率: PASS")
    else:
        print("❌ 參數效率: FAIL")
    
    if test_results.get('inference_speed', False):
        print("✅ 推理速度: PASS")
    else:
        print("❌ 推理速度: FAIL")
    
    # 推薦配置
    if 'param_details' in test_results:
        print("\n🎯 推薦配置:")
        best_configs = [r for r in test_results['param_details'] if r['parameters'] < 1e6]
        if best_configs:
            best_config = max(best_configs, key=lambda x: x['out_channels'])
            print(f"  🏆 最佳: {best_config['name']} ({best_config['parameters']:,} 參數)")
        else:
            print("  ⚠️ 所有配置都超過 1M 參數限制")
    
    # 最終狀態
    if passed_tests == total_tests:
        print("\n🎉 所有測試通過！頸部網路就緒。")
        return True
    else:
        print(f"\n⚠️ {total_tests - passed_tests} 個測試失敗，需要修復。")
        return False


if __name__ == "__main__":
    # 設置 CUDA 設備
    if torch.cuda.device_count() > 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    print("🏗️ 頸部網路測試腳本")
    print(f"📅 測試時間: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python: {sys.version}")
    print(f"🔥 PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"🚀 CUDA: {torch.version.cuda}")
        print(f"📱 GPU: {torch.cuda.get_device_name()}")
    else:
        print("💻 使用 CPU")
    
    print("\n" + "=" * 60)
    
    # 運行測試
    results = run_comprehensive_tests()
    
    # 打印總結
    success = print_final_summary(results)
    
    # 退出碼
    sys.exit(0 if success else 1)
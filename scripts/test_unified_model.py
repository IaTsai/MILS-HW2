#!/usr/bin/env python3
"""
完整統一多任務模型測試腳本
測試骨幹網路 + 頸部網路 + 多任務頭部的完整模型
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
from src.models.neck import create_neck
from src.models.head import create_multitask_head
from src.models.unified_model import create_unified_model, UnifiedMultiTaskModel


def test_model_components():
    """測試模型各組件參數量"""
    print("🧪 測試模型各組件參數量...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 測試骨幹網路
    backbone = create_backbone('mobilenetv3_small', pretrained=False)
    backbone_params = backbone.get_parameter_count()
    print(f"🔧 骨幹網路: {backbone_params['total_parameters']:,} 參數 ({backbone_params['total_parameters']/1e6:.2f}M)")
    
    # 2. 測試頸部網路
    neck = create_neck('fpn', in_channels_list=[16, 24, 48, 96], out_channels=128)
    neck_params = neck.get_parameter_count()
    print(f"🔗 頸部網路: {neck_params['total_parameters']:,} 參數 ({neck_params['total_parameters']/1e6:.2f}M)")
    
    # 3. 測試頭部網路
    head = create_multitask_head(
        'unified',
        in_channels=128,
        num_det_classes=10,
        num_seg_classes=21,
        num_cls_classes=10,
        shared_channels=256
    )
    head_params = head.get_parameter_count()
    print(f"🎯 頭部網路: {head_params['total_parameters']:,} 參數 ({head_params['total_parameters']/1e6:.2f}M)")
    
    # 總計
    total_estimated = backbone_params['total_parameters'] + neck_params['total_parameters'] + head_params['total_parameters']
    print(f"📊 預估總計: {total_estimated:,} 參數 ({total_estimated/1e6:.2f}M)")
    
    return {
        'backbone': backbone_params['total_parameters'],
        'neck': neck_params['total_parameters'],
        'head': head_params['total_parameters'],
        'total_estimated': total_estimated
    }


def test_unified_model_creation():
    """測試統一模型創建"""
    print("\n🏗️ 測試統一模型創建...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 測試不同配置
    configs = ['default', 'lightweight']
    
    results = {}
    
    for config_name in configs:
        print(f"\n📋 測試配置: {config_name}")
        
        try:
            model = create_unified_model(config_name)
            model = model.to(device)
            
            # 獲取參數統計
            param_info = model.get_total_parameters()
            model_info = model.get_model_info()
            
            print(f"  ✅ 模型創建成功")
            print(f"  📊 總參數: {param_info['total_parameters']:,} ({param_info['total_parameters']/1e6:.2f}M)")
            print(f"  🎯 預算使用率: {model_info['parameter_budget']['utilization_rate']:.1%}")
            
            # 檢查參數預算
            budget_ok = param_info['total_parameters'] <= 8_000_000
            print(f"  {'✅' if budget_ok else '❌'} 參數預算: {budget_ok}")
            
            results[config_name] = {
                'success': True,
                'parameters': param_info['total_parameters'],
                'budget_ok': budget_ok,
                'model_info': model_info
            }
            
        except Exception as e:
            print(f"  ❌ 模型創建失敗: {e}")
            results[config_name] = {
                'success': False,
                'error': str(e)
            }
    
    return results


def test_model_forward_pass():
    """測試模型前向傳播"""
    print("\n🚀 測試模型前向傳播...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 創建模型
    model = create_unified_model('default')
    model = model.to(device)
    model.eval()
    
    # 測試不同輸入尺寸
    test_cases = [
        (1, 224, 224),
        (2, 320, 320),
        (4, 512, 512)
    ]
    
    results = {}
    
    for batch_size, height, width in test_cases:
        print(f"\n🧪 測試輸入: ({batch_size}, 3, {height}, {width})")
        
        try:
            # 創建測試輸入
            test_input = torch.randn(batch_size, 3, height, width).to(device)
            
            # 測試完整推理
            with torch.no_grad():
                start_time = time.time()
                outputs = model(test_input, task_type='all')
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
            
            inference_time = (end_time - start_time) * 1000  # ms
            
            print(f"  ✅ 前向傳播成功")
            print(f"  ⏱️ 推理時間: {inference_time:.2f}ms")
            print(f"  🔍 輸出形狀:")
            
            for task, output in outputs.items():
                print(f"    {task}: {output.shape}")
            
            # 測試單任務推理
            single_task_times = {}
            for task in ['detection', 'segmentation', 'classification']:
                with torch.no_grad():
                    start_time = time.time()
                    single_output = model(test_input, task_type=task)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    end_time = time.time()
                
                single_task_times[task] = (end_time - start_time) * 1000
            
            print(f"  🎯 單任務推理時間:")
            for task, task_time in single_task_times.items():
                print(f"    {task}: {task_time:.2f}ms")
            
            results[f"{batch_size}x{height}x{width}"] = {
                'success': True,
                'inference_time_ms': inference_time,
                'single_task_times': single_task_times,
                'output_shapes': {task: list(output.shape) for task, output in outputs.items()},
                'meets_latency': inference_time < 150.0
            }
            
        except Exception as e:
            print(f"  ❌ 前向傳播失敗: {e}")
            results[f"{batch_size}x{height}x{width}"] = {
                'success': False,
                'error': str(e)
            }
    
    return results


def test_model_benchmark():
    """測試模型性能基準"""
    print("\n⏱️ 執行模型性能基準測試...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 創建模型
    model = create_unified_model('default')
    model = model.to(device)
    
    # 執行基準測試
    try:
        benchmark_results = model.inference_benchmark(
            input_size=(512, 512),
            batch_size=1,
            num_warmup=10,
            num_runs=50
        )
        
        print("✅ 性能基準測試成功！")
        print("📈 測試結果:")
        
        for key, value in benchmark_results.items():
            if isinstance(value, (int, float)):
                if 'time' in key or 'fps' in key:
                    print(f"  {key}: {value:.3f}")
                elif 'memory' in key:
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")
        
        return benchmark_results
        
    except Exception as e:
        print(f"❌ 性能基準測試失敗: {e}")
        return {'success': False, 'error': str(e)}


def test_model_save_load():
    """測試模型保存和加載"""
    print("\n💾 測試模型保存和加載...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # 創建原始模型
        original_model = create_unified_model('default')
        original_model = original_model.to(device)
        
        # 保存模型
        save_path = '/tmp/test_unified_model.pth'
        original_model.save_model(save_path, epoch=0, test_mode=True)
        
        # 加載模型
        loaded_model, loaded_info = UnifiedMultiTaskModel.load_model(save_path, device=device)
        
        # 驗證加載的模型
        test_input = torch.randn(1, 3, 224, 224).to(device)
        
        with torch.no_grad():
            original_output = original_model(test_input)
            loaded_output = loaded_model(test_input)
        
        # 檢查輸出是否一致
        outputs_match = True
        for task in original_output:
            if not torch.allclose(original_output[task], loaded_output[task], atol=1e-5):
                outputs_match = False
                break
        
        print(f"✅ 模型保存和加載成功！")
        print(f"🔍 輸出一致性: {'✅ 一致' if outputs_match else '❌ 不一致'}")
        
        # 清理
        os.remove(save_path)
        
        return {'success': True, 'outputs_match': outputs_match}
        
    except Exception as e:
        print(f"❌ 模型保存/加載失敗: {e}")
        return {'success': False, 'error': str(e)}


def run_comprehensive_test():
    """運行全面測試"""
    print("🚀 開始統一多任務模型全面測試...")
    print("=" * 70)
    
    results = {}
    
    # 1. 組件參數測試
    try:
        component_results = test_model_components()
        results['components'] = component_results
    except Exception as e:
        print(f"❌ 組件測試失敗: {e}")
        results['components'] = {'error': str(e)}
    
    # 2. 模型創建測試
    try:
        creation_results = test_unified_model_creation()
        results['creation'] = creation_results
    except Exception as e:
        print(f"❌ 模型創建測試失敗: {e}")
        results['creation'] = {'error': str(e)}
    
    # 3. 前向傳播測試
    try:
        forward_results = test_model_forward_pass()
        results['forward'] = forward_results
    except Exception as e:
        print(f"❌ 前向傳播測試失敗: {e}")
        results['forward'] = {'error': str(e)}
    
    # 4. 性能基準測試
    try:
        benchmark_results = test_model_benchmark()
        results['benchmark'] = benchmark_results
    except Exception as e:
        print(f"❌ 性能基準測試失敗: {e}")
        results['benchmark'] = {'error': str(e)}
    
    # 5. 保存/加載測試
    try:
        save_load_results = test_model_save_load()
        results['save_load'] = save_load_results
    except Exception as e:
        print(f"❌ 保存/加載測試失敗: {e}")
        results['save_load'] = {'error': str(e)}
    
    return results


def print_final_summary(results):
    """打印最終測試總結"""
    print("\n" + "=" * 70)
    print("📋 統一多任務模型測試總結")
    print("=" * 70)
    
    # 成功率統計
    total_tests = len(results)
    successful_tests = sum(1 for r in results.values() if isinstance(r, dict) and 'error' not in r)
    
    print(f"✅ 測試通過: {successful_tests}/{total_tests}")
    
    # 關鍵指標總結
    if 'creation' in results and 'default' in results['creation']:
        default_info = results['creation']['default']
        if 'model_info' in default_info:
            model_info = default_info['model_info']
            params = model_info['parameters']
            budget = model_info['parameter_budget']
            
            print(f"\n🏗️ 默認配置模型:")
            print(f"  總參數: {params['total_parameters']:,} ({params['total_parameters']/1e6:.2f}M)")
            print(f"  預算使用: {budget['utilization_rate']:.1%}")
            print(f"  預算檢查: {'✅ 通過' if budget['used_parameters'] <= budget['total_budget'] else '❌ 超出'}")
    
    # 性能指標
    if 'benchmark' in results and 'avg_inference_time_ms' in results['benchmark']:
        benchmark = results['benchmark']
        print(f"\n⏱️ 性能指標 (512x512):")
        print(f"  推理時間: {benchmark['avg_inference_time_ms']:.2f}ms")
        print(f"  FPS: {benchmark['fps']:.1f}")
        print(f"  延遲要求: {'✅ 滿足' if benchmark['meets_latency_requirement'] else '❌ 不滿足'} (<150ms)")
        
        if 'avg_memory_usage_mb' in benchmark:
            print(f"  記憶體使用: {benchmark['avg_memory_usage_mb']:.1f}MB")
    
    # 輸出格式驗證
    if 'forward' in results:
        forward_results = results['forward']
        successful_forward = sum(1 for r in forward_results.values() if r.get('success', False))
        print(f"\n🔍 前向傳播測試: {successful_forward}/{len(forward_results)} 成功")
        
        # 檢查輸出格式
        for size, result in forward_results.items():
            if result.get('success', False):
                shapes = result['output_shapes']
                print(f"  {size}:")
                for task, shape in shapes.items():
                    print(f"    {task}: {shape}")
                break
    
    # 最終結論
    print(f"\n🎯 最終結論:")
    
    # 檢查關鍵要求
    requirements_met = []
    
    # 1. 參數量檢查
    param_ok = False
    if 'creation' in results and 'default' in results['creation']:
        param_ok = results['creation']['default'].get('budget_ok', False)
    requirements_met.append(('參數量 < 8M', param_ok))
    
    # 2. 推理速度檢查
    speed_ok = False
    if 'benchmark' in results:
        speed_ok = results['benchmark'].get('meets_latency_requirement', False)
    requirements_met.append(('推理速度 < 150ms', speed_ok))
    
    # 3. 前向傳播檢查
    forward_ok = False
    if 'forward' in results:
        forward_ok = any(r.get('success', False) for r in results['forward'].values())
    requirements_met.append(('前向傳播正常', forward_ok))
    
    # 4. 模型保存/加載檢查
    save_load_ok = False
    if 'save_load' in results:
        save_load_ok = results['save_load'].get('success', False)
    requirements_met.append(('模型保存/加載', save_load_ok))
    
    for requirement, met in requirements_met:
        print(f"  {'✅' if met else '❌'} {requirement}")
    
    all_requirements_met = all(met for _, met in requirements_met)
    
    if all_requirements_met:
        print(f"\n🎉 所有要求都已滿足！模型準備就緒。")
        print(f"✅ Phase 2 完成！準備進入 Phase 3: 防遺忘策略實現")
        return True
    else:
        failed_count = sum(1 for _, met in requirements_met if not met)
        print(f"\n⚠️ {failed_count} 個要求未滿足，需要修復。")
        return False


if __name__ == "__main__":
    print("🏗️ 統一多任務模型測試腳本")
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
    test_results = run_comprehensive_test()
    
    # 打印總結
    success = print_final_summary(test_results)
    
    # 退出碼
    sys.exit(0 if success else 1)
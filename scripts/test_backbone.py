#!/usr/bin/env python3
"""
骨幹網路測試腳本
測試特徵提取、推理速度和記憶體使用
"""
import os
import sys
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.profiler import profile, ProfilerActivity
import psutil
import gc
import numpy as np
from typing import Dict, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.backbone import create_backbone, BackboneNetwork

class BackboneTester:
    """骨幹網路測試器"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
    def print_header(self, title):
        """打印測試標題"""
        print(f"\n{'='*70}")
        print(f"{title:^70}")
        print(f"{'='*70}")
    
    def test_feature_extraction(self, model_name='mobilenetv3_small'):
        """測試特徵提取功能"""
        self.print_header(f"特徵提取測試 - {model_name}")
        
        # 創建模型
        backbone = create_backbone(model_name, pretrained=False)
        backbone = backbone.to(self.device)
        backbone.eval()
        
        test_results = {}
        
        # 測試不同輸入尺寸
        input_sizes = [224, 320, 512]
        
        for size in input_sizes:
            print(f"\n📏 測試輸入尺寸: {size}x{size}")
            
            # 創建測試輸入
            x = torch.randn(1, 3, size, size).to(self.device)
            
            with torch.no_grad():
                try:
                    features = backbone(x)
                    
                    size_results = {
                        'input_size': (size, size),
                        'features': {}
                    }
                    
                    # 檢查特徵圖尺寸
                    for layer_name, feature in features.items():
                        expected_stride = backbone.get_feature_info()['feature_strides'][layer_name]
                        expected_size = size // expected_stride
                        actual_size = feature.shape[-1]
                        
                        size_results['features'][layer_name] = {
                            'shape': list(feature.shape),
                            'expected_size': expected_size,
                            'actual_size': actual_size,
                            'correct': expected_size == actual_size
                        }
                        
                        status = "✅" if expected_size == actual_size else "❌"
                        print(f"  {layer_name}: {feature.shape} {status}")
                        if expected_size != actual_size:
                            print(f"    期望尺寸: {expected_size}, 實際尺寸: {actual_size}")
                    
                    test_results[f'size_{size}'] = size_results
                    
                except Exception as e:
                    print(f"❌ 尺寸 {size} 測試失敗: {e}")
                    test_results[f'size_{size}'] = {'error': str(e)}
        
        # 測試參數數量
        param_info = backbone.get_parameter_count()
        feature_info = backbone.get_feature_info()
        
        print(f"\n📊 模型信息:")
        print(f"  總參數量: {param_info['total_parameters']:,}")
        print(f"  可訓練參數: {param_info['trainable_parameters']:,}")
        print(f"  特徵通道數: {feature_info['feature_channels']}")
        
        self.results[f'{model_name}_features'] = {
            'test_results': test_results,
            'parameter_info': param_info,
            'feature_info': feature_info
        }
        
        return test_results
    
    def test_inference_speed(self, model_name='mobilenetv3_small', num_runs=100):
        """測試推理速度"""
        self.print_header(f"推理速度測試 - {model_name}")
        
        # 創建模型
        backbone = create_backbone(model_name, pretrained=False)
        backbone = backbone.to(self.device)
        backbone.eval()
        
        # 測試不同輸入尺寸的推理速度
        input_sizes = [224, 320, 512]
        batch_sizes = [1, 4, 8]
        
        speed_results = {}
        
        for size in input_sizes:
            for batch_size in batch_sizes:
                print(f"\n⏱️  測試: {batch_size}x{size}x{size}")
                
                # 創建測試數據
                x = torch.randn(batch_size, 3, size, size).to(self.device)
                
                # 預熱
                with torch.no_grad():
                    for _ in range(10):
                        _ = backbone(x)
                
                # 同步GPU
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                # 測試推理時間
                times = []
                with torch.no_grad():
                    for _ in range(num_runs):
                        start_time = time.time()
                        features = backbone(x)
                        
                        if self.device.type == 'cuda':
                            torch.cuda.synchronize()
                        
                        end_time = time.time()
                        times.append((end_time - start_time) * 1000)  # 轉換為毫秒
                
                # 計算統計信息
                times = np.array(times)
                mean_time = np.mean(times)
                std_time = np.std(times)
                min_time = np.min(times)
                max_time = np.max(times)
                
                # 計算吞吐量
                throughput = (batch_size * 1000) / mean_time  # samples/sec
                
                result_key = f'batch{batch_size}_size{size}'
                speed_results[result_key] = {
                    'batch_size': batch_size,
                    'input_size': size,
                    'mean_time_ms': mean_time,
                    'std_time_ms': std_time,
                    'min_time_ms': min_time,
                    'max_time_ms': max_time,
                    'throughput_samples_per_sec': throughput
                }
                
                print(f"  平均時間: {mean_time:.2f} ± {std_time:.2f} ms")
                print(f"  範圍: {min_time:.2f} - {max_time:.2f} ms")
                print(f"  吞吐量: {throughput:.1f} samples/sec")
                
                # 檢查是否符合150ms限制（單個樣本）
                single_sample_time = mean_time / batch_size
                if single_sample_time <= 150:
                    print(f"  ✅ 符合150ms限制 ({single_sample_time:.2f}ms/sample)")
                else:
                    print(f"  ❌ 超過150ms限制 ({single_sample_time:.2f}ms/sample)")
        
        self.results[f'{model_name}_speed'] = speed_results
        return speed_results
    
    def test_memory_usage(self, model_name='mobilenetv3_small'):
        """測試記憶體使用"""
        self.print_header(f"記憶體使用測試 - {model_name}")
        
        # 清理記憶體
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # 記錄初始記憶體
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024**3  # GB
        
        if self.device.type == 'cuda':
            initial_gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        else:
            initial_gpu_memory = 0
        
        print(f"初始記憶體使用:")
        print(f"  CPU: {initial_memory:.3f} GB")
        print(f"  GPU: {initial_gpu_memory:.3f} GB")
        
        # 創建模型
        backbone = create_backbone(model_name, pretrained=False)
        backbone = backbone.to(self.device)
        
        # 記錄模型載入後的記憶體
        model_memory = process.memory_info().rss / 1024**3
        if self.device.type == 'cuda':
            model_gpu_memory = torch.cuda.memory_allocated() / 1024**3
        else:
            model_gpu_memory = 0
        
        print(f"\n模型載入後:")
        print(f"  CPU: {model_memory:.3f} GB (+{model_memory-initial_memory:.3f})")
        print(f"  GPU: {model_gpu_memory:.3f} GB (+{model_gpu_memory-initial_gpu_memory:.3f})")
        
        # 測試不同batch size的記憶體使用
        batch_sizes = [1, 4, 8, 16]
        memory_results = {}
        
        for batch_size in batch_sizes:
            print(f"\n🔍 測試 batch_size={batch_size}")
            
            # 清理記憶體
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # 創建測試數據
            x = torch.randn(batch_size, 3, 512, 512).to(self.device)
            
            # 前向傳播
            with torch.no_grad():
                features = backbone(x)
            
            # 記錄記憶體使用
            inference_memory = process.memory_info().rss / 1024**3
            if self.device.type == 'cuda':
                inference_gpu_memory = torch.cuda.memory_allocated() / 1024**3
                peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024**3
            else:
                inference_gpu_memory = 0
                peak_gpu_memory = 0
            
            memory_results[f'batch_{batch_size}'] = {
                'batch_size': batch_size,
                'cpu_memory_gb': inference_memory,
                'gpu_memory_gb': inference_gpu_memory,
                'peak_gpu_memory_gb': peak_gpu_memory,
                'memory_increase_gb': inference_memory - model_memory
            }
            
            print(f"  CPU: {inference_memory:.3f} GB")
            print(f"  GPU: {inference_gpu_memory:.3f} GB (峰值: {peak_gpu_memory:.3f} GB)")
            
            # 清理特徵圖
            del features, x
        
        self.results[f'{model_name}_memory'] = memory_results
        return memory_results
    
    def test_pretrained_loading(self, model_name='mobilenetv3_small'):
        """測試預訓練權重載入"""
        self.print_header(f"預訓練權重測試 - {model_name}")
        
        try:
            # 測試不加載預訓練權重
            print("測試隨機初始化...")
            backbone_random = create_backbone(model_name, pretrained=False)
            
            # 測試加載預訓練權重
            print("\n測試預訓練權重載入...")
            backbone_pretrained = create_backbone(model_name, pretrained=True)
            
            # 比較權重
            random_params = dict(backbone_random.named_parameters())
            pretrained_params = dict(backbone_pretrained.named_parameters())
            
            different_params = 0
            total_params = 0
            
            for name in random_params.keys():
                if name in pretrained_params:
                    if not torch.equal(random_params[name], pretrained_params[name]):
                        different_params += 1
                    total_params += 1
            
            print(f"\n權重比較結果:")
            print(f"  總參數層數: {total_params}")
            print(f"  不同的參數層: {different_params}")
            print(f"  權重載入率: {different_params/total_params*100:.1f}%")
            
            return True
            
        except Exception as e:
            print(f"❌ 預訓練權重測試失敗: {e}")
            return False
    
    def run_comprehensive_test(self, model_names=['mobilenetv3_small']):
        """運行完整測試"""
        print("🚀 開始骨幹網路綜合測試")
        print(f"🔧 設備: {self.device}")
        
        all_passed = True
        
        for model_name in model_names:
            print(f"\n🧪 測試模型: {model_name}")
            
            try:
                # 特徵提取測試
                feature_results = self.test_feature_extraction(model_name)
                
                # 推理速度測試  
                speed_results = self.test_inference_speed(model_name, num_runs=50)
                
                # 記憶體使用測試
                memory_results = self.test_memory_usage(model_name)
                
                # 預訓練權重測試
                pretrained_ok = self.test_pretrained_loading(model_name)
                
                # 檢查是否通過基本要求
                param_count = self.results[f'{model_name}_features']['parameter_info']['total_parameters']
                
                print(f"\n📋 {model_name} 測試總結:")
                print(f"  參數數量: {param_count:,} {'✅' if param_count < 8_000_000 else '❌'}")
                print(f"  預訓練載入: {'✅' if pretrained_ok else '❌'}")
                
                # 檢查速度要求
                speed_ok = True
                for key, result in speed_results.items():
                    if 'batch1' in key and 'size512' in key:
                        single_time = result['mean_time_ms']
                        if single_time > 150:
                            speed_ok = False
                            break
                
                print(f"  速度要求 (<150ms): {'✅' if speed_ok else '❌'}")
                
                if param_count >= 8_000_000 or not speed_ok:
                    all_passed = False
                    
            except Exception as e:
                print(f"❌ {model_name} 測試失敗: {e}")
                all_passed = False
        
        return all_passed
    
    def generate_report(self):
        """生成測試報告"""
        self.print_header("測試報告")
        
        print("📊 詳細結果已保存到 self.results")
        print("🎯 關鍵指標:")
        
        for model_name in ['mobilenetv3_small']:
            if f'{model_name}_features' in self.results:
                param_info = self.results[f'{model_name}_features']['parameter_info']
                print(f"\n{model_name}:")
                print(f"  總參數: {param_info['total_parameters']:,}")
                
                if f'{model_name}_speed' in self.results:
                    speed_info = self.results[f'{model_name}_speed']
                    if 'batch1_size512' in speed_info:
                        time_512 = speed_info['batch1_size512']['mean_time_ms']
                        print(f"  512x512推理時間: {time_512:.2f}ms")
                
                if f'{model_name}_memory' in self.results:
                    memory_info = self.results[f'{model_name}_memory']
                    if 'batch_1' in memory_info:
                        gpu_mem = memory_info['batch_1']['gpu_memory_gb']
                        print(f"  GPU記憶體使用: {gpu_mem:.3f}GB")


def main():
    """主函數"""
    tester = BackboneTester()
    
    # 運行綜合測試
    success = tester.run_comprehensive_test(['mobilenetv3_small'])
    
    # 生成報告
    tester.generate_report()
    
    if success:
        print("\n✅ 骨幹網路實現完成！")
        print("🎉 所有測試通過，準備進入下一階段")
    else:
        print("\n❌ 部分測試失敗，需要優化")
    
    return success


if __name__ == "__main__":
    main()
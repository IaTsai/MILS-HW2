"""
統一多任務學習模型
整合骨幹網路、頸部網路、多任務頭部的完整模型實現
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple, Any
import time

from .backbone import create_backbone, BackboneNetwork
from .neck import create_neck, FeaturePyramidNetwork
from .head import create_multitask_head, UnifiedMultiTaskHead


class UnifiedMultiTaskModel(nn.Module):
    """
    統一多任務學習模型
    
    整合骨幹網路、頸部網路、多任務頭部的完整架構：
    Input -> Backbone -> Neck -> Head -> Multi-task Outputs
    
    支援：
    - 同時進行檢測、分割、分類三個任務
    - 單任務推理模式
    - 靈活的參數配置
    - 高效的特徵共享
    
    Args:
        backbone_name: 骨幹網路名稱
        neck_type: 頸部網路類型
        head_type: 頭部網路類型
        num_det_classes: 檢測類別數
        num_seg_classes: 分割類別數
        num_cls_classes: 分類類別數
        pretrained: 是否使用預訓練權重
    """
    
    def __init__(self,
                 backbone_name: str = 'mobilenetv3_small',
                 neck_type: str = 'fpn',
                 head_type: str = 'unified',
                 num_det_classes: int = 10,
                 num_seg_classes: int = 21,
                 num_cls_classes: int = 10,
                 pretrained: bool = True,
                 neck_out_channels: int = 128,
                 head_shared_channels: int = 256):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.neck_type = neck_type
        self.head_type = head_type
        self.num_det_classes = num_det_classes
        self.num_seg_classes = num_seg_classes
        self.num_cls_classes = num_cls_classes
        
        # 創建骨幹網路
        self.backbone = create_backbone(
            model_name=backbone_name,
            pretrained=pretrained
        )
        
        # 獲取骨幹網路特徵通道配置
        backbone_channels = self.backbone.feature_channels
        in_channels_list = [
            backbone_channels['layer1'],
            backbone_channels['layer2'], 
            backbone_channels['layer3'],
            backbone_channels['layer4']
        ]
        
        # 創建頸部網路
        if neck_type == 'fpn':
            self.neck = create_neck(
                neck_type='fpn',
                in_channels_list=in_channels_list,
                out_channels=neck_out_channels
            )
        elif neck_type == 'lightweight':
            self.neck = create_neck(
                neck_type='lightweight',
                in_channels_list=in_channels_list,
                out_channels=neck_out_channels
            )
        else:
            raise ValueError(f"Unsupported neck type: {neck_type}")
        
        # 創建多任務頭部
        if head_type == 'unified':
            self.head = create_multitask_head(
                head_type='unified',
                in_channels=neck_out_channels,
                num_det_classes=num_det_classes,
                num_seg_classes=num_seg_classes,
                num_cls_classes=num_cls_classes,
                shared_channels=head_shared_channels
            )
        else:
            raise ValueError(f"Unsupported head type: {head_type}")
        
        # 模型信息
        self.model_info = self._get_model_info()
    
    def forward(self, 
                x: torch.Tensor, 
                task_type: str = 'all') -> Dict[str, torch.Tensor]:
        """
        前向傳播
        
        Args:
            x: 輸入圖像 (B, 3, H, W)
            task_type: 任務類型 'all', 'detection', 'segmentation', 'classification'
        
        Returns:
            outputs: 任務輸出字典
        """
        # 骨幹網路特徵提取
        backbone_features = self.backbone(x)
        
        # 頸部網路特徵融合
        neck_features = self.neck(backbone_features)
        
        # 多任務頭部輸出
        head_outputs = self.head(neck_features, task_type=task_type)
        
        return head_outputs
    
    def get_total_parameters(self) -> Dict[str, int]:
        """獲取模型總參數量統計"""
        # 骨幹網路參數
        backbone_params = self.backbone.get_parameter_count()
        
        # 頸部網路參數
        neck_params = self.neck.get_parameter_count()
        
        # 頭部網路參數
        head_params = self.head.get_parameter_count()
        
        # 總參數量
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'backbone_parameters': backbone_params['total_parameters'],
            'neck_parameters': neck_params['total_parameters'],
            'head_parameters': head_params['total_parameters'],
            'backbone_detail': backbone_params,
            'neck_detail': neck_params,
            'head_detail': head_params
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """獲取模型詳細信息"""
        return self.model_info
    
    def _get_model_info(self) -> Dict[str, Any]:
        """計算模型信息"""
        param_info = self.get_total_parameters()
        
        return {
            'architecture': {
                'backbone': self.backbone_name,
                'neck': self.neck_type,
                'head': self.head_type
            },
            'task_config': {
                'detection_classes': self.num_det_classes,
                'segmentation_classes': self.num_seg_classes,
                'classification_classes': self.num_cls_classes
            },
            'parameters': param_info,
            'parameter_budget': {
                'total_budget': 8_000_000,
                'used_parameters': param_info['total_parameters'],
                'remaining_budget': 8_000_000 - param_info['total_parameters'],
                'utilization_rate': param_info['total_parameters'] / 8_000_000
            }
        }
    
    def inference_benchmark(self, 
                           input_size: Tuple[int, int] = (512, 512),
                           batch_size: int = 1,
                           num_warmup: int = 10,
                           num_runs: int = 50,
                           device: Optional[torch.device] = None) -> Dict[str, float]:
        """
        推理性能基準測試
        
        Args:
            input_size: 輸入圖像尺寸
            batch_size: 批次大小
            num_warmup: 熱身次數
            num_runs: 測試次數
            device: 計算設備
        
        Returns:
            benchmark_results: 性能測試結果
        """
        if device is None:
            device = next(self.parameters()).device
        
        self.eval()
        
        # 創建測試輸入
        test_input = torch.randn(batch_size, 3, input_size[0], input_size[1]).to(device)
        
        # 熱身
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = self(test_input, task_type='all')
        
        # 性能測試
        times = []
        memory_usage = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                # 記錄起始記憶體
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                    start_memory = torch.cuda.memory_allocated()
                
                # 計時
                start_time = time.time()
                outputs = self(test_input, task_type='all')
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.time()
                inference_time = (end_time - start_time) * 1000  # 轉換為毫秒
                times.append(inference_time)
                
                # 記錄記憶體使用
                if device.type == 'cuda':
                    peak_memory = torch.cuda.max_memory_allocated()
                    memory_usage.append((peak_memory - start_memory) / 1024**2)  # MB
        
        # 計算統計信息
        import numpy as np
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        results = {
            'input_size': input_size,
            'batch_size': batch_size,
            'avg_inference_time_ms': avg_time,
            'std_inference_time_ms': std_time,
            'min_inference_time_ms': min_time,
            'max_inference_time_ms': max_time,
            'fps': 1000 / avg_time,
            'meets_latency_requirement': avg_time < 150.0  # < 150ms 要求
        }
        
        if device.type == 'cuda' and memory_usage:
            results.update({
                'avg_memory_usage_mb': np.mean(memory_usage),
                'peak_memory_usage_mb': np.max(memory_usage)
            })
        
        return results
    
    def save_model(self, save_path: str, include_optimizer: bool = False, **kwargs):
        """保存模型"""
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'backbone_name': self.backbone_name,
                'neck_type': self.neck_type,
                'head_type': self.head_type,
                'num_det_classes': self.num_det_classes,
                'num_seg_classes': self.num_seg_classes,
                'num_cls_classes': self.num_cls_classes
            },
            'model_info': self.model_info,
            **kwargs
        }
        
        torch.save(save_dict, save_path)
        print(f"✅ 模型已保存到: {save_path}")
    
    @classmethod
    def load_model(cls, load_path: str, device: Optional[torch.device] = None):
        """加載模型"""
        checkpoint = torch.load(load_path, map_location=device)
        
        # 重建模型
        model_config = checkpoint['model_config']
        model = cls(**model_config)
        
        # 加載權重
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if device is not None:
            model = model.to(device)
        
        print(f"✅ 模型已從 {load_path} 加載")
        return model, checkpoint.get('model_info', {})
    
    def print_model_summary(self):
        """打印模型總結"""
        info = self.get_model_info()
        
        print("🏗️ 統一多任務學習模型總結")
        print("=" * 60)
        
        # 架構信息
        arch = info['architecture']
        print(f"🔧 架構配置:")
        print(f"  骨幹網路: {arch['backbone']}")
        print(f"  頸部網路: {arch['neck']}")
        print(f"  頭部網路: {arch['head']}")
        
        # 任務配置
        task = info['task_config']
        print(f"\n🎯 任務配置:")
        print(f"  檢測類別: {task['detection_classes']}")
        print(f"  分割類別: {task['segmentation_classes']}")
        print(f"  分類類別: {task['classification_classes']}")
        
        # 參數統計
        params = info['parameters']
        budget = info['parameter_budget']
        
        print(f"\n📊 參數統計:")
        print(f"  總參數量: {params['total_parameters']:,} ({params['total_parameters']/1e6:.2f}M)")
        print(f"  骨幹網路: {params['backbone_parameters']:,} ({params['backbone_parameters']/1e6:.2f}M)")
        print(f"  頸部網路: {params['neck_parameters']:,} ({params['neck_parameters']/1e6:.2f}M)")
        print(f"  頭部網路: {params['head_parameters']:,} ({params['head_parameters']/1e6:.2f}M)")
        
        print(f"\n💰 參數預算:")
        print(f"  預算限制: {budget['total_budget']:,} (8.0M)")
        print(f"  已使用: {budget['used_parameters']:,}")
        print(f"  剩餘: {budget['remaining_budget']:,}")
        print(f"  使用率: {budget['utilization_rate']:.1%}")
        
        # 預算檢查
        if budget['used_parameters'] <= budget['total_budget']:
            print(f"  ✅ 參數量符合預算限制")
        else:
            print(f"  ❌ 參數量超出預算限制")


def create_unified_model(config_name: str = 'default', **kwargs) -> UnifiedMultiTaskModel:
    """
    創建統一多任務模型的工廠函數
    
    Args:
        config_name: 配置名稱
        **kwargs: 覆蓋配置參數
    
    Returns:
        model: 統一多任務模型實例
    """
    # 預定義配置
    configs = {
        'default': {
            'backbone_name': 'mobilenetv3_small',
            'neck_type': 'fpn',
            'head_type': 'unified',
            'num_det_classes': 10,
            'num_seg_classes': 21,
            'num_cls_classes': 10,
            'pretrained': True,
            'neck_out_channels': 128,
            'head_shared_channels': 256
        },
        'lightweight': {
            'backbone_name': 'mobilenetv3_small',
            'neck_type': 'lightweight',
            'head_type': 'unified',
            'num_det_classes': 10,
            'num_seg_classes': 21,
            'num_cls_classes': 10,
            'pretrained': True,
            'neck_out_channels': 64,
            'head_shared_channels': 128
        },
        'high_performance': {
            'backbone_name': 'mobilenetv3_small',
            'neck_type': 'fpn',
            'head_type': 'unified',
            'num_det_classes': 10,
            'num_seg_classes': 21,
            'num_cls_classes': 10,
            'pretrained': True,
            'neck_out_channels': 256,
            'head_shared_channels': 512
        }
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")
    
    config = configs[config_name].copy()
    config.update(kwargs)
    
    return UnifiedMultiTaskModel(**config)


if __name__ == "__main__":
    # 測試代碼
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("🚀 測試統一多任務學習模型")
    print("=" * 60)
    
    # 創建模型
    model = create_unified_model('default')
    model = model.to(device)
    
    # 打印模型總結
    model.print_model_summary()
    
    # 測試前向傳播
    print("\n🧪 測試前向傳播...")
    batch_size = 2
    test_input = torch.randn(batch_size, 3, 512, 512).to(device)
    
    with torch.no_grad():
        outputs = model(test_input, task_type='all')
    
    print("✅ 前向傳播測試成功！")
    print("\n🔍 輸出形狀:")
    for task, output in outputs.items():
        print(f"  {task}: {output.shape}")
    
    # 性能基準測試
    print("\n⏱️ 執行性能基準測試...")
    benchmark_results = model.inference_benchmark(
        input_size=(512, 512),
        batch_size=1,
        num_warmup=5,
        num_runs=20
    )
    
    print("📈 性能測試結果:")
    for key, value in benchmark_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\n🎉 模型測試完成！")
    if benchmark_results['meets_latency_requirement']:
        print("✅ 滿足推理速度要求 (<150ms)")
    else:
        print("❌ 不滿足推理速度要求 (<150ms)")
    
    # 檢查參數預算
    info = model.get_model_info()
    budget = info['parameter_budget']
    if budget['used_parameters'] <= budget['total_budget']:
        print("✅ 參數量符合預算限制")
    else:
        print("❌ 參數量超出預算限制")
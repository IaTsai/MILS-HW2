"""
頸部網路實現 - Feature Pyramid Network (FPN)
用於多尺度特徵融合，支援多任務學習
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network (FPN) 頸部網路
    
    實現多尺度特徵融合，將骨幹網路的不同層級特徵
    融合成統一的特徵金字塔，適用於多任務學習
    
    Args:
        in_channels_list: 輸入特徵圖的通道數列表 [16, 24, 48, 96]
        out_channels: 輸出特徵圖的統一通道數，預設256
        extra_blocks: 是否添加額外的下採樣塊
    """
    
    def __init__(self, 
                 in_channels_list: List[int] = [16, 24, 48, 96], 
                 out_channels: int = 256,
                 extra_blocks: Optional[int] = None):
        super().__init__()
        
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.num_feature_levels = len(in_channels_list)
        
        # 橫向連接層 (1x1 卷積降維)
        # 將不同通道數的特徵圖統一到 out_channels
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            lateral_conv = nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=1, 
                stride=1, 
                padding=0
            )
            self.lateral_convs.append(lateral_conv)
        
        # 融合卷積層 (3x3 卷積消除混疊)
        # 對上採樣後的特徵進行平滑處理
        self.fpn_convs = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            fpn_conv = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
            self.fpn_convs.append(fpn_conv)
        
        # 額外的下採樣塊 (用於更大感受野)
        self.extra_blocks = None
        if extra_blocks:
            self.extra_blocks = nn.ModuleList()
            for i in range(extra_blocks):
                extra_conv = nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1
                )
                self.extra_blocks.append(extra_conv)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化網路權重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向傳播
        
        Args:
            features: 骨幹網路輸出的多尺度特徵
                     {'layer1': tensor, 'layer2': tensor, 'layer3': tensor, 'layer4': tensor}
        
        Returns:
            fused_features: 融合後的特徵金字塔
                           {'P2': tensor, 'P3': tensor, 'P4': tensor, 'P5': tensor}
        """
        # 提取特徵並按順序排列 (從小尺度到大尺度)
        feature_list = []
        layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
        
        for layer_name in layer_names:
            if layer_name in features:
                feature_list.append(features[layer_name])
            else:
                raise KeyError(f"Missing feature layer: {layer_name}")
        
        # 自頂向下路徑 (Top-down pathway)
        # 從最小尺度特徵開始
        laterals = []
        
        # 1. 通過橫向連接處理所有特徵
        for i, feature in enumerate(feature_list):
            lateral = self.lateral_convs[i](feature)
            laterals.append(lateral)
        
        # 2. 自頂向下融合
        # 從最高層(最小解析度)開始
        fpn_features = []
        
        # 最高層特徵直接使用
        prev_feature = laterals[-1]
        fpn_features.append(self.fpn_convs[-1](prev_feature))
        
        # 自頂向下融合其他層
        for i in range(len(laterals) - 2, -1, -1):
            # 上採樣到當前層的大小
            upsampled = F.interpolate(
                prev_feature, 
                size=laterals[i].shape[-2:], 
                mode='nearest'
            )
            
            # 與橫向連接相加
            fused = upsampled + laterals[i]
            
            # 通過 3x3 卷積平滑
            fpn_feature = self.fpn_convs[i](fused)
            fpn_features.append(fpn_feature)
            
            prev_feature = fused
        
        # 反轉列表，使其按照從大尺度到小尺度排列
        fpn_features = fpn_features[::-1]
        
        # 3. 構建輸出字典
        output_features = {}
        pyramid_levels = ['P2', 'P3', 'P4', 'P5']
        
        for i, fpn_feature in enumerate(fpn_features):
            output_features[pyramid_levels[i]] = fpn_feature
        
        # 4. 添加額外的下採樣特徵 (如果有)
        if self.extra_blocks:
            last_feature = fpn_features[-1]
            for i, extra_conv in enumerate(self.extra_blocks):
                last_feature = extra_conv(last_feature)
                output_features[f'P{6+i}'] = last_feature
        
        return output_features
    
    def get_parameter_count(self):
        """獲取參數數量統計"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # 詳細統計
        lateral_params = sum(p.numel() for conv in self.lateral_convs for p in conv.parameters())
        fpn_params = sum(p.numel() for conv in self.fpn_convs for p in conv.parameters())
        extra_params = 0
        if self.extra_blocks:
            extra_params = sum(p.numel() for conv in self.extra_blocks for p in conv.parameters())
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'lateral_conv_parameters': lateral_params,
            'fpn_conv_parameters': fpn_params,
            'extra_block_parameters': extra_params
        }
    
    def get_feature_info(self):
        """獲取特徵信息"""
        return {
            'input_channels': self.in_channels_list,
            'output_channels': self.out_channels,
            'num_feature_levels': self.num_feature_levels,
            'has_extra_blocks': self.extra_blocks is not None,
            'output_strides': {
                'P2': 4,   # 1/4 resolution
                'P3': 8,   # 1/8 resolution
                'P4': 16,  # 1/16 resolution
                'P5': 32   # 1/32 resolution
            }
        }


class LightweightNeck(nn.Module):
    """
    輕量化頸部網路 (替代方案)
    
    使用更簡單的特徵融合策略，參數量更少
    適用於資源受限的環境
    
    Args:
        in_channels_list: 輸入特徵圖的通道數列表
        out_channels: 輸出特徵圖的統一通道數
        use_bn: 是否使用 BatchNorm
    """
    
    def __init__(self, 
                 in_channels_list: List[int] = [16, 24, 48, 96],
                 out_channels: int = 128,
                 use_bn: bool = True):
        super().__init__()
        
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.use_bn = use_bn
        
        # 簡單的 1x1 卷積統一通道數
        self.channel_reducers = nn.ModuleList()
        for in_channels in in_channels_list:
            layers = [nn.Conv2d(in_channels, out_channels, 1, 1, 0)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            
            self.channel_reducers.append(nn.Sequential(*layers))
        
        # 可選的融合卷積
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化權重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """前向傳播"""
        fused_features = {}
        layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
        output_names = ['P2', 'P3', 'P4', 'P5']
        
        for i, (layer_name, output_name) in enumerate(zip(layer_names, output_names)):
            if layer_name in features:
                # 通道數統一
                reduced_feature = self.channel_reducers[i](features[layer_name])
                # 可選的融合卷積
                fused_feature = self.fusion_conv(reduced_feature)
                fused_features[output_name] = fused_feature
            else:
                raise KeyError(f"Missing feature layer: {layer_name}")
        
        return fused_features
    
    def get_parameter_count(self):
        """獲取參數數量統計"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }


# 向後兼容性
class FPN(nn.Module):
    """原始 FPN 類，保持向後兼容"""
    def __init__(self, in_channels, out_channel=256):
        super(FPN, self).__init__()
        
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_channel in in_channels:
            l_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
            fpn_conv = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        
        self.num_ins = len(in_channels)
    
    def forward(self, inputs):
        assert len(inputs) == self.num_ins
        
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(inputs[i]))
        
        for i in range(self.num_ins - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], 
                size=laterals[i - 1].shape[2:], 
                mode='nearest'
            )
        
        outs = []
        for i, fpn_conv in enumerate(self.fpn_convs):
            outs.append(fpn_conv(laterals[i]))
        
        return tuple(outs)


def create_neck(neck_type='fpn', **kwargs):
    """
    頸部網路工廠函數
    
    Args:
        neck_type: 網路類型 'fpn' 或 'lightweight'
        **kwargs: 其他參數
    
    Returns:
        neck: 頸部網路實例
    """
    if neck_type == 'fpn':
        return FeaturePyramidNetwork(**kwargs)
    elif neck_type == 'lightweight':
        return LightweightNeck(**kwargs)
    else:
        raise ValueError(f"Unsupported neck type: {neck_type}")


if __name__ == "__main__":
    # 測試代碼
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    
    # 輸出結果
    print("✅ FPN 頸部網路測試成功！")
    print(f"📊 參數統計: {fpn.get_parameter_count()}")
    print(f"📋 特徵信息: {fpn.get_feature_info()}")
    
    print("\n🔍 輸出特徵形狀:")
    for name, feature in output_features.items():
        print(f"  {name}: {feature.shape}")
    
    # 測試輕量化版本
    print("\n" + "="*50)
    lightweight_neck = create_neck('lightweight', in_channels_list=[16, 24, 48, 96], out_channels=128)
    lightweight_neck = lightweight_neck.to(device)
    
    with torch.no_grad():
        lightweight_output = lightweight_neck(input_features)
    
    print("✅ 輕量化頸部網路測試成功！")
    print(f"📊 參數統計: {lightweight_neck.get_parameter_count()}")
    
    print("\n🔍 輸出特徵形狀:")
    for name, feature in lightweight_output.items():
        print(f"  {name}: {feature.shape}")
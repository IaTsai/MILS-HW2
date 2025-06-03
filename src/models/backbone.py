"""
骨幹網路實現
支援多種輕量化網路架構，專為多任務學習設計
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.models as tv_models
from typing import Dict, List, Optional, Tuple
import warnings

class MobileNetV3Block(nn.Module):
    """MobileNetV3 基本塊"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, 
                 se_ratio=0.25, activation='relu', use_se=True):
        super().__init__()
        
        self.use_residual = stride == 1 and in_channels == out_channels
        hidden_dim = int(in_channels * expand_ratio)
        
        layers = []
        
        # 1x1 pointwise conv (expand)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                self._get_activation(activation)
            ])
        
        # depthwise conv
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 
                     kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            self._get_activation(activation)
        ])
        
        # SE block
        if use_se:
            layers.append(SqueezeExcite(hidden_dim, int(hidden_dim * se_ratio)))
        
        # 1x1 pointwise conv (project)
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def _get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'hswish':
            return HardSwish()
        else:
            return nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.conv(x)
        if self.use_residual:
            out = out + x
        return out


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excite 注意力模塊"""
    
    def __init__(self, in_channels, reduced_dim):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Hardsigmoid(inplace=True)
        )
    
    def forward(self, x):
        return x * self.se(x)


class HardSwish(nn.Module):
    """Hard Swish 激活函數"""
    
    def forward(self, x):
        return x * F.relu6(x + 3.0) / 6.0


class MobileNetV3Small(nn.Module):
    """MobileNetV3-Small 骨幹網路實現"""
    
    def __init__(self, num_classes=1000, dropout=0.2):
        super().__init__()
        
        # 第一層卷積
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            HardSwish()
        )
        
        # MobileNetV3-Small 配置
        # [in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, activation]
        configs = [
            [16, 16, 3, 2, 1, 0.25, 'relu'],      # layer1
            [16, 24, 3, 2, 4.5, None, 'relu'],   # layer2
            [24, 24, 3, 1, 3.67, None, 'relu'],
            [24, 40, 5, 2, 4, 0.25, 'hswish'],   # layer3  
            [40, 40, 5, 1, 6, 0.25, 'hswish'],
            [40, 40, 5, 1, 6, 0.25, 'hswish'],
            [40, 48, 5, 1, 3, 0.25, 'hswish'],
            [48, 48, 5, 1, 3, 0.25, 'hswish'],
            [48, 96, 5, 2, 6, 0.25, 'hswish'],   # layer4
            [96, 96, 5, 1, 6, 0.25, 'hswish'],
            [96, 96, 5, 1, 6, 0.25, 'hswish'],
        ]
        
        # 構建特徵提取層
        self.features = nn.ModuleList()
        for i, (in_c, out_c, k, s, e, se, act) in enumerate(configs):
            use_se = se is not None
            se_ratio = se if se is not None else 0.25
            self.features.append(
                MobileNetV3Block(in_c, out_c, k, s, e, se_ratio, act, use_se)
            )
        
        # 分類頭 (用於預訓練)
        self.classifier = nn.Sequential(
            nn.Conv2d(96, 576, 1, 1, 0, bias=False),
            nn.BatchNorm2d(576),
            HardSwish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(576, 1024, 1, 1, 0),
            HardSwish(),
            nn.Dropout(dropout),
            nn.Conv2d(1024, num_classes, 1, 1, 0)
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
    
    def forward(self, x, return_features=False):
        """前向傳播"""
        x = self.conv1(x)  # 1/2
        
        # 提取多尺度特徵
        features = {}
        
        # Layer 1: 1/4 resolution  
        x = self.features[0](x)
        features['layer1'] = x
        
        # Layer 2: 1/8 resolution
        for i in range(1, 3):
            x = self.features[i](x)
        features['layer2'] = x
        
        # Layer 3: 1/16 resolution
        for i in range(3, 8):
            x = self.features[i](x)
        features['layer3'] = x
        
        # Layer 4: 1/32 resolution
        for i in range(8, 11):
            x = self.features[i](x)
        features['layer4'] = x
        
        if return_features:
            return features
        
        # 分類輸出
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        
        return x


class BackboneNetwork(nn.Module):
    """統一的骨幹網路接口"""
    
    def __init__(self, model_name='mobilenetv3_small', pretrained=True, freeze_bn=False):
        super().__init__()
        
        self.model_name = model_name
        self.freeze_bn = freeze_bn
        
        if model_name == 'mobilenetv3_small':
            self.backbone = self._build_mobilenetv3_small(pretrained)
            self.feature_channels = {
                'layer1': 16,   # 1/4 resolution
                'layer2': 24,   # 1/8 resolution  
                'layer3': 48,   # 1/16 resolution (實際輸出通道數)
                'layer4': 96    # 1/32 resolution
            }
        elif model_name == 'efficientnet_b0':
            self.backbone = self._build_efficientnet_b0(pretrained)
            self.feature_channels = {
                'layer1': 24,   # 1/4 resolution
                'layer2': 40,   # 1/8 resolution
                'layer3': 112,  # 1/16 resolution
                'layer4': 320   # 1/32 resolution
            }
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        if freeze_bn:
            self._freeze_bn()
    
    def _build_mobilenetv3_small(self, pretrained):
        """構建 MobileNetV3-Small"""
        model = MobileNetV3Small()
        
        if pretrained:
            try:
                # 嘗試加載 torchvision 預訓練權重
                pretrained_model = tv_models.mobilenet_v3_small(pretrained=True)
                self._load_pretrained_weights(model, pretrained_model)
                print("✅ 成功加載 MobileNetV3-Small 預訓練權重")
            except Exception as e:
                warnings.warn(f"無法加載預訓練權重: {e}")
                print("⚠️ 使用隨機初始化權重")
        
        return model
    
    def _build_efficientnet_b0(self, pretrained):
        """構建 EfficientNet-B0 (備選方案)"""
        try:
            if pretrained:
                model = tv_models.efficientnet_b0(pretrained=True)
            else:
                model = tv_models.efficientnet_b0(pretrained=False)
            print("✅ 成功創建 EfficientNet-B0")
            return model
        except Exception as e:
            raise RuntimeError(f"無法創建 EfficientNet-B0: {e}")
    
    def _load_pretrained_weights(self, model, pretrained_model):
        """加載預訓練權重（部分匹配）"""
        model_dict = model.state_dict()
        pretrained_dict = pretrained_model.state_dict()
        
        # 過濾匹配的權重
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and model_dict[k].shape == v.shape}
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        print(f"加載了 {len(pretrained_dict)} 個預訓練參數")
    
    def _freeze_bn(self):
        """凍結 BatchNorm 層"""
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
    
    def forward(self, x):
        """前向傳播，返回多尺度特徵"""
        if self.model_name == 'mobilenetv3_small':
            return self.backbone(x, return_features=True)
        elif self.model_name == 'efficientnet_b0':
            return self._extract_efficientnet_features(x)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
    
    def _extract_efficientnet_features(self, x):
        """從 EfficientNet 提取多尺度特徵"""
        features = {}
        
        # EfficientNet features extraction
        x = self.backbone.features[0](x)  # stem
        
        # Layer 1: after block 2 (1/4)
        for i in range(1, 3):
            x = self.backbone.features[i](x)
        features['layer1'] = x
        
        # Layer 2: after block 4 (1/8)
        for i in range(3, 5):
            x = self.backbone.features[i](x)
        features['layer2'] = x
        
        # Layer 3: after block 6 (1/16)
        for i in range(5, 7):
            x = self.backbone.features[i](x)
        features['layer3'] = x
        
        # Layer 4: final features (1/32)
        for i in range(7, len(self.backbone.features)):
            x = self.backbone.features[i](x)
        features['layer4'] = x
        
        return features
    
    def get_parameter_count(self):
        """獲取參數數量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params
        }
    
    def get_feature_info(self):
        """獲取特徵信息"""
        return {
            'model_name': self.model_name,
            'feature_channels': self.feature_channels,
            'feature_strides': {
                'layer1': 4,
                'layer2': 8, 
                'layer3': 16,
                'layer4': 32
            }
        }
    
    def train(self, mode=True):
        """重寫 train 方法以處理凍結的 BN"""
        super().train(mode)
        if self.freeze_bn:
            self._freeze_bn()
        return self


def create_backbone(model_name='mobilenetv3_small', pretrained=True, **kwargs):
    """創建骨幹網路的工廠函數"""
    return BackboneNetwork(model_name=model_name, pretrained=pretrained, **kwargs)


# 保持向後兼容性
class Backbone(BackboneNetwork):
    """向後兼容的 Backbone 類"""
    def __init__(self, backbone_name='mobilenetv3_small', pretrained=True):
        super().__init__(model_name=backbone_name, pretrained=pretrained)


if __name__ == "__main__":
    # 測試代碼
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 創建模型
    backbone = create_backbone('mobilenetv3_small', pretrained=False)
    backbone = backbone.to(device)
    
    # 測試輸入
    x = torch.randn(1, 3, 512, 512).to(device)
    
    # 前向傳播
    with torch.no_grad():
        features = backbone(x)
    
    # 打印結果
    print("✅ 骨幹網路測試成功！")
    print(f"模型參數: {backbone.get_parameter_count()}")
    print(f"特徵信息: {backbone.get_feature_info()}")
    
    for name, feat in features.items():
        print(f"{name}: {feat.shape}")
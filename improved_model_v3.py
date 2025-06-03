import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.mobilenetv3 import mobilenet_v3_small
from typing import Dict, List, Optional, Tuple

class ImprovedBackbone(nn.Module):
    """MobileNetV3-Small backbone with enhanced features"""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        # Load MobileNetV3-Small
        mobilenet = mobilenet_v3_small(pretrained=pretrained)
        
        # Extract features without the classifier
        self.features = mobilenet.features
        
        # Get output channels for each stage
        self.stage_channels = []
        for idx in [3, 5, 8, 11]:  # Key stages in MobileNetV3-Small
            out_channels = self.features[idx].out_channels
            self.stage_channels.append(out_channels)
            
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        
        for idx, module in enumerate(self.features):
            x = module(x)
            if idx in [3, 5, 8, 11]:  # Extract at key stages
                features.append(x)
                
        return features

class ImprovedNeck(nn.Module):
    """FPN neck with 128 channels"""
    def __init__(self, in_channels: List[int], out_channels: int = 128):
        super().__init__()
        self.out_channels = out_channels
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1)
            for in_ch in in_channels
        ])
        
        # Output convolutions
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=False)  # No inplace for gradient safety
            ) for _ in in_channels
        ])
        
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # Lateral connections
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        
        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            shape = laterals[i-1].shape[-2:]
            laterals[i-1] = laterals[i-1] + F.interpolate(
                laterals[i], size=shape, mode='nearest'
            )
        
        # Output convolutions
        outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        
        return outs

class EnhancedClassificationHead(nn.Module):
    """
    Enhanced classification head without BatchNorm
    - 移除BatchNorm解決多任務訓練不穩定問題
    - 增加容量和表達能力
    - 使用LayerNorm替代BatchNorm
    """
    def __init__(self, in_channels: int = 128, num_classes: int = 10, 
                 hidden_dims: List[int] = [512, 256, 128], dropout_rate: float = 0.3):
        super().__init__()
        
        # Multi-scale pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Calculate input dimension (avg + max pooling)
        pool_dim = in_channels * 2
        
        # Build classification layers without BatchNorm
        layers = []
        prev_dim = pool_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # 使用LayerNorm替代BatchNorm
                nn.ReLU(inplace=False),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
            
        # Final classification layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        # Better initialization
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # Use the highest resolution feature map
        x = features[0]
        
        # Multi-scale pooling
        avg_feat = self.avg_pool(x).flatten(1)
        max_feat = self.max_pool(x).flatten(1)
        x = torch.cat([avg_feat, max_feat], dim=1)
        
        # Classification
        return self.classifier(x)

class ImprovedSegmentationHead(nn.Module):
    """Segmentation head with enhanced decoder"""
    def __init__(self, in_channels: int = 128, num_classes: int = 21):
        super().__init__()
        
        # Decoder blocks
        self.decoder = nn.Sequential(
            # First upsampling block
            nn.ConvTranspose2d(in_channels, 256, 4, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=False),
            
            # Second upsampling block
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=False),
            
            # Third upsampling block
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=False),
            
            # Final upsampling block
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=False),
            
            # Output layer
            nn.Conv2d(32, num_classes, 1)
        )
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # Use the highest resolution feature
        x = features[0]
        
        # Decode to full resolution
        x = self.decoder(x)
        
        # Ensure output size matches input (512x512)
        x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
        
        return x

class ImprovedDetectionHead(nn.Module):
    """FCOS-style detection head"""
    def __init__(self, in_channels: int = 128, num_classes: int = 10):
        super().__init__()
        
        # Shared convolutions
        self.shared_convs = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=False),
        )
        
        # Task-specific heads
        self.bbox_pred = nn.Conv2d(256, 4, 3, padding=1)
        self.cls_pred = nn.Conv2d(256, num_classes, 3, padding=1)
        self.centerness_pred = nn.Conv2d(256, 1, 3, padding=1)
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        outputs = []
        
        for feature in features:
            shared_feat = self.shared_convs(feature)
            
            bbox = self.bbox_pred(shared_feat)
            cls = self.cls_pred(shared_feat)
            centerness = self.centerness_pred(shared_feat)
            
            # Reshape and concatenate
            B, _, H, W = bbox.shape
            output = torch.cat([
                bbox.permute(0, 2, 3, 1).reshape(B, -1, 4),
                cls.permute(0, 2, 3, 1).reshape(B, -1, 10),
                centerness.permute(0, 2, 3, 1).reshape(B, -1, 1)
            ], dim=-1)
            
            outputs.append(output)
            
        return torch.cat(outputs, dim=1)

class UnifiedModelV3(nn.Module):
    """
    第三版統一模型：專注於解決災難性遺忘
    主要改進：
    1. 移除分類頭部的BatchNorm
    2. 使用LayerNorm替代
    3. 增強分類頭部容量
    4. 更好的初始化策略
    """
    def __init__(self, 
                 num_classes: Dict[str, int] = {'classification': 10, 'segmentation': 21, 'detection': 10},
                 classification_hidden_dims: List[int] = [512, 256, 128],
                 classification_dropout: float = 0.3):
        super().__init__()
        
        # Shared backbone
        self.backbone = ImprovedBackbone(pretrained=True)
        
        # Neck
        self.neck = ImprovedNeck(self.backbone.stage_channels, out_channels=128)
        
        # Task-specific heads
        self.classification_head = EnhancedClassificationHead(
            in_channels=128,
            num_classes=num_classes['classification'],
            hidden_dims=classification_hidden_dims,
            dropout_rate=classification_dropout
        )
        
        self.segmentation_head = ImprovedSegmentationHead(
            in_channels=128,
            num_classes=num_classes['segmentation']
        )
        
        self.detection_head = ImprovedDetectionHead(
            in_channels=128,
            num_classes=num_classes['detection']
        )
        
    def forward(self, x: torch.Tensor, task: str) -> torch.Tensor:
        # Extract features
        features = self.backbone(x)
        
        # Apply neck
        neck_features = self.neck(features)
        
        # Task-specific forward
        if task == 'classification':
            return self.classification_head(neck_features)
        elif task == 'segmentation':
            return self.segmentation_head(neck_features)
        elif task == 'detection':
            return self.detection_head(neck_features)
        else:
            raise ValueError(f"Unknown task: {task}")
            
    def get_params_by_task(self, task: str) -> List[torch.nn.Parameter]:
        """Get parameters for a specific task"""
        if task == 'classification':
            return list(self.classification_head.parameters())
        elif task == 'segmentation':
            return list(self.segmentation_head.parameters())
        elif task == 'detection':
            return list(self.detection_head.parameters())
        elif task == 'backbone':
            return list(self.backbone.parameters()) + list(self.neck.parameters())
        else:
            raise ValueError(f"Unknown task: {task}")
            
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters for each component"""
        counts = {
            'backbone': sum(p.numel() for p in self.backbone.parameters()),
            'neck': sum(p.numel() for p in self.neck.parameters()),
            'classification': sum(p.numel() for p in self.classification_head.parameters()),
            'segmentation': sum(p.numel() for p in self.segmentation_head.parameters()),
            'detection': sum(p.numel() for p in self.detection_head.parameters()),
        }
        counts['total'] = sum(counts.values())
        return counts


if __name__ == "__main__":
    # Test the model
    model = UnifiedModelV3()
    
    # Print parameter counts
    param_counts = model.count_parameters()
    print("🔧 Model V3 參數統計:")
    for name, count in param_counts.items():
        print(f"  {name}: {count:,} ({count/1e6:.2f}M)")
        
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 3, 512, 512)
    
    # Test each task
    for task in ['classification', 'segmentation', 'detection']:
        output = model(x, task)
        print(f"\n{task} output shape: {output.shape}")
        
    # Verify total parameters < 8M
    total_params = param_counts['total']
    print(f"\n✅ 總參數量: {total_params:,} ({total_params/1e6:.2f}M) < 8M: {total_params < 8e6}")
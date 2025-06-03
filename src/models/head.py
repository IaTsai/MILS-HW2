"""
統一多任務頭部實現
用於同時處理檢測、分割、分類三個任務的統一頭部設計
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple


class UnifiedMultiTaskHead(nn.Module):
    """
    統一多任務頭部 - 核心設計
    
    採用共享特徵提取 + 任務特定分支的設計策略：
    - 2層共享卷積特徵提取 (參數高效)
    - 3個任務特定輸出分支 (保持任務特異性)
    - 支援多尺度特徵融合
    
    Args:
        in_channels: 輸入特徵通道數 (來自FPN)
        num_det_classes: 檢測類別數
        num_seg_classes: 分割類別數  
        num_cls_classes: 分類類別數
        shared_channels: 共享特徵通道數
    """
    
    def __init__(self, 
                 in_channels: int = 128,
                 num_det_classes: int = 10,
                 num_seg_classes: int = 21, 
                 num_cls_classes: int = 10,
                 shared_channels: int = 256):
        super().__init__()
        
        self.in_channels = in_channels
        self.shared_channels = shared_channels
        self.num_det_classes = num_det_classes
        self.num_seg_classes = num_seg_classes
        self.num_cls_classes = num_cls_classes
        
        # 共享特徵提取層 (2層)
        self.shared_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, shared_channels, 3, 1, 1),
            nn.BatchNorm2d(shared_channels),
            nn.ReLU(inplace=True)
        )
        
        self.shared_conv2 = nn.Sequential(
            nn.Conv2d(shared_channels, shared_channels, 3, 1, 1),
            nn.BatchNorm2d(shared_channels),
            nn.ReLU(inplace=True)
        )
        
        # 檢測分支 (anchor-free FCOS-style)
        self.detection_cls = nn.Conv2d(shared_channels, num_det_classes, 3, 1, 1)
        self.detection_reg = nn.Conv2d(shared_channels, 4, 3, 1, 1)  # l, t, r, b
        self.detection_centerness = nn.Conv2d(shared_channels, 1, 3, 1, 1)
        
        # 分割分支 (輕量化UNet-style)
        self.segmentation_conv = nn.Sequential(
            nn.Conv2d(shared_channels, shared_channels // 2, 3, 1, 1),
            nn.BatchNorm2d(shared_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_channels // 2, num_seg_classes, 1, 1, 0)
        )
        
        # 分類分支 (全局平均池化)
        self.classification_gap = nn.AdaptiveAvgPool2d(1)
        self.classification_fc = nn.Sequential(
            nn.Linear(shared_channels, shared_channels // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(shared_channels // 4, num_cls_classes)
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
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # 特殊初始化檢測分支的bias
        nn.init.constant_(self.detection_cls.bias, -2.19)  # 對應0.1的先驗概率
    
    def forward(self, features: Dict[str, torch.Tensor], 
                task_type: str = 'all') -> Dict[str, torch.Tensor]:
        """
        前向傳播
        
        Args:
            features: FPN輸出的多尺度特徵 {'P2': tensor, 'P3': tensor, 'P4': tensor, 'P5': tensor}
            task_type: 任務類型 'all', 'detection', 'segmentation', 'classification'
        
        Returns:
            outputs: 任務輸出字典
        """
        # 選擇主要特徵層 (P3: 1/8 resolution, 平衡分辨率和語義)
        if 'P3' in features:
            main_feature = features['P3']
        elif 'P2' in features:
            main_feature = features['P2']
        else:
            # 使用第一個可用特徵
            main_feature = list(features.values())[0]
        
        # 共享特徵提取
        shared_feat = self.shared_conv1(main_feature)
        shared_feat = self.shared_conv2(shared_feat)
        
        outputs = {}
        
        # 檢測任務
        if task_type in ['all', 'detection']:
            det_cls = self.detection_cls(shared_feat)
            det_reg = self.detection_reg(shared_feat)
            det_centerness = self.detection_centerness(shared_feat)
            
            # 轉換為標準格式: (B, H*W, 6) - cx,cy,w,h,conf,cls
            B, _, H, W = det_cls.shape
            
            # 分類logits: (B, num_classes, H, W) -> (B, H*W, num_classes)
            det_cls = det_cls.permute(0, 2, 3, 1).reshape(B, H*W, self.num_det_classes)
            
            # 回歸: (B, 4, H, W) -> (B, H*W, 4)
            det_reg = det_reg.permute(0, 2, 3, 1).reshape(B, H*W, 4)
            det_reg = F.relu(det_reg)  # 確保正值
            
            # 中心度: (B, 1, H, W) -> (B, H*W, 1)
            det_centerness = det_centerness.permute(0, 2, 3, 1).reshape(B, H*W, 1)
            det_centerness = torch.sigmoid(det_centerness)
            
            # 組合輸出: cx, cy, w, h, centerness, class_logits
            # 生成網格座標
            device = det_cls.device
            y_coords, x_coords = torch.meshgrid(
                torch.arange(H, device=device, dtype=torch.float32),
                torch.arange(W, device=device, dtype=torch.float32),
                indexing='ij'
            )
            
            # 座標歸一化到[0,1]
            x_coords = (x_coords + 0.5) / W
            y_coords = (y_coords + 0.5) / H
            
            coords = torch.stack([x_coords, y_coords], dim=-1)  # (H, W, 2)
            coords = coords.reshape(H*W, 2).unsqueeze(0).expand(B, -1, -1)  # (B, H*W, 2)
            
            # FCOS格式轉換為中心點格式
            # l, t, r, b -> cx, cy, w, h
            l, t, r, b = det_reg[..., 0:1], det_reg[..., 1:2], det_reg[..., 2:3], det_reg[..., 3:4]
            
            cx = coords[..., 0:1] - l + (l + r) / 2
            cy = coords[..., 1:2] - t + (t + b) / 2
            w = l + r
            h = t + b
            
            # 找出最高置信度類別
            max_cls_scores, max_cls_indices = torch.max(torch.sigmoid(det_cls), dim=-1, keepdim=True)
            
            # 綜合置信度 (分類置信度 * 中心度)
            confidence = max_cls_scores * det_centerness
            
            # 最終檢測輸出: (B, H*W, 6)
            detection_output = torch.cat([cx, cy, w, h, confidence, max_cls_indices.float()], dim=-1)
            
            outputs['detection'] = detection_output
        
        # 分割任務
        if task_type in ['all', 'segmentation']:
            seg_out = self.segmentation_conv(shared_feat)
            
            # 上採樣到原圖尺寸 (假設輸入為512x512, P3為64x64)
            target_size = (512, 512)  # 根據實際輸入尺寸調整
            seg_out = F.interpolate(
                seg_out, 
                size=target_size, 
                mode='bilinear', 
                align_corners=False
            )
            
            outputs['segmentation'] = seg_out  # (B, num_seg_classes, H, W)
        
        # 分類任務
        if task_type in ['all', 'classification']:
            cls_feat = self.classification_gap(shared_feat)  # (B, C, 1, 1)
            cls_feat = cls_feat.flatten(1)  # (B, C)
            cls_out = self.classification_fc(cls_feat)
            
            outputs['classification'] = cls_out  # (B, num_cls_classes)
        
        return outputs
    
    def get_parameter_count(self):
        """獲取參數數量統計"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # 分組統計
        shared_params = (
            sum(p.numel() for p in self.shared_conv1.parameters()) + 
            sum(p.numel() for p in self.shared_conv2.parameters())
        )
        
        detection_params = (
            sum(p.numel() for p in self.detection_cls.parameters()) +
            sum(p.numel() for p in self.detection_reg.parameters()) +
            sum(p.numel() for p in self.detection_centerness.parameters())
        )
        
        segmentation_params = sum(p.numel() for p in self.segmentation_conv.parameters())
        
        classification_params = sum(p.numel() for p in self.classification_fc.parameters())
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'shared_parameters': shared_params,
            'detection_parameters': detection_params,
            'segmentation_parameters': segmentation_params,
            'classification_parameters': classification_params
        }
    
    def get_output_info(self):
        """獲取輸出信息"""
        return {
            'detection_format': 'FCOS-style (cx, cy, w, h, confidence, class)',
            'detection_shape': '(B, H*W, 6)',
            'segmentation_format': 'Dense prediction',
            'segmentation_shape': '(B, num_seg_classes, H, W)',
            'classification_format': 'Logits',
            'classification_shape': '(B, num_cls_classes)',
            'num_det_classes': self.num_det_classes,
            'num_seg_classes': self.num_seg_classes,
            'num_cls_classes': self.num_cls_classes
        }


# 向後兼容的單任務頭部類
class DetectionHead(nn.Module):
    """檢測頭部 (向後兼容)"""
    def __init__(self, in_channels, num_classes, num_anchors=9):
        super(DetectionHead, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        self.cls_head = nn.Conv2d(
            in_channels, 
            num_anchors * num_classes, 
            kernel_size=3, 
            padding=1
        )
        self.reg_head = nn.Conv2d(
            in_channels, 
            num_anchors * 4, 
            kernel_size=3, 
            padding=1
        )
    
    def forward(self, x):
        cls_out = self.cls_head(x)
        reg_out = self.reg_head(x)
        return cls_out, reg_out


class SegmentationHead(nn.Module):
    """分割頭部 (向後兼容)"""
    def __init__(self, in_channels, num_classes, feature_scale=4):
        super(SegmentationHead, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        
        self.conv3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.feature_scale = feature_scale
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = F.interpolate(x, scale_factor=self.feature_scale, mode='bilinear', align_corners=True)
        
        return x


class ClassificationHead(nn.Module):
    """分類頭部 (向後兼容)"""
    def __init__(self, in_channels, num_classes):
        super(ClassificationHead, self).__init__()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)
    
    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def create_multitask_head(head_type='unified', **kwargs):
    """
    多任務頭部工廠函數
    
    Args:
        head_type: 頭部類型 'unified' 或 'separate'
        **kwargs: 其他參數
    
    Returns:
        head: 頭部網路實例
    """
    if head_type == 'unified':
        return UnifiedMultiTaskHead(**kwargs)
    elif head_type == 'separate':
        # 返回分離的頭部字典
        return {
            'detection': DetectionHead(**kwargs.get('detection', {})),
            'segmentation': SegmentationHead(**kwargs.get('segmentation', {})),
            'classification': ClassificationHead(**kwargs.get('classification', {}))
        }
    else:
        raise ValueError(f"Unsupported head type: {head_type}")


if __name__ == "__main__":
    # 測試代碼
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 創建統一多任務頭部
    head = UnifiedMultiTaskHead(
        in_channels=128,
        num_det_classes=10,
        num_seg_classes=21,
        num_cls_classes=10,
        shared_channels=256
    )
    head = head.to(device)
    
    # 模擬FPN輸出
    batch_size = 2
    fpn_features = {
        'P2': torch.randn(batch_size, 128, 128, 128).to(device),
        'P3': torch.randn(batch_size, 128, 64, 64).to(device),
        'P4': torch.randn(batch_size, 128, 32, 32).to(device),
        'P5': torch.randn(batch_size, 128, 16, 16).to(device)
    }
    
    # 前向傳播測試
    with torch.no_grad():
        outputs = head(fpn_features, task_type='all')
    
    print("✅ 統一多任務頭部測試成功！")
    print(f"📊 參數統計: {head.get_parameter_count()}")
    print(f"📋 輸出信息: {head.get_output_info()}")
    
    print("\n🔍 輸出形狀:")
    for task, output in outputs.items():
        print(f"  {task}: {output.shape}")
    
    # 測試單任務推理
    print("\n🧪 測試單任務推理:")
    for task in ['detection', 'segmentation', 'classification']:
        with torch.no_grad():
            single_output = head(fpn_features, task_type=task)
        print(f"  {task}: {single_output[task].shape}")
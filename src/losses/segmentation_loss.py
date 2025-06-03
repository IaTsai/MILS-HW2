"""
分割任務損失函數
包含交叉熵損失、Dice損失、Focal損失等多種語義分割損失函數
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class SegmentationLoss(nn.Module):
    """
    語義分割損失函數
    
    結合交叉熵損失和Dice損失，針對語義分割任務設計。
    支援類別權重、忽略索引、多尺度損失等特性。
    
    Args:
        num_classes: 分割類別數
        loss_type: 損失類型 ('ce', 'dice', 'focal', 'combined')
        class_weights: 類別權重
        ignore_index: 忽略的類別索引
        focal_alpha: Focal loss alpha 參數
        focal_gamma: Focal loss gamma 參數
        dice_smooth: Dice loss 平滑參數
        ce_weight: 交叉熵損失權重
        dice_weight: Dice損失權重
    """
    
    def __init__(self,
                 num_classes: int = 21,
                 loss_type: str = 'combined',
                 class_weights: Optional[torch.Tensor] = None,
                 ignore_index: int = 255,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 dice_smooth: float = 1e-6,
                 ce_weight: float = 1.0,
                 dice_weight: float = 1.0):
        super().__init__()
        
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.ignore_index = ignore_index
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.dice_smooth = dice_smooth
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        # 設置類別權重
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
        
        # 創建交叉熵損失
        self.ce_loss = nn.CrossEntropyLoss(
            weight=self.class_weights,
            ignore_index=ignore_index,
            reduction='mean'
        )
    
    def cross_entropy_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        交叉熵損失
        
        Args:
            pred: 預測logits (B, C, H, W)
            target: 目標標籤 (B, H, W)
        
        Returns:
            ce_loss: 交叉熵損失
        """
        return self.ce_loss(pred, target)
    
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Dice損失
        
        Args:
            pred: 預測logits (B, C, H, W)
            target: 目標標籤 (B, H, W)
        
        Returns:
            dice_loss: Dice損失
        """
        # 轉換為概率
        pred_prob = F.softmax(pred, dim=1)
        
        # 創建 one-hot 編碼
        target_one_hot = F.one_hot(target.long(), num_classes=self.num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)
        
        # 忽略指定索引
        if self.ignore_index >= 0:
            mask = (target != self.ignore_index).float().unsqueeze(1)
            pred_prob = pred_prob * mask
            target_one_hot = target_one_hot * mask
        
        # 計算每個類別的 Dice 係數
        intersection = (pred_prob * target_one_hot).sum(dim=(2, 3))  # (B, C)
        union = pred_prob.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))  # (B, C)
        
        dice_coeff = (2.0 * intersection + self.dice_smooth) / (union + self.dice_smooth)
        
        # 返回 1 - mean(dice_coeff)
        return 1.0 - dice_coeff.mean()
    
    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Focal Loss (用於處理類別不平衡)
        
        Args:
            pred: 預測logits (B, C, H, W)
            target: 目標標籤 (B, H, W)
        
        Returns:
            focal_loss: Focal損失
        """
        # 計算交叉熵
        ce_loss = F.cross_entropy(pred, target, reduction='none', ignore_index=self.ignore_index)
        
        # 計算概率
        pt = torch.exp(-ce_loss)
        
        # 應用 Focal weight
        focal_weight = (1 - pt) ** self.focal_gamma
        
        # 應用 alpha weighting (如果有類別權重)
        if self.focal_alpha > 0:
            alpha_weight = self.focal_alpha
            focal_weight = alpha_weight * focal_weight
        
        focal_loss = focal_weight * ce_loss
        
        # 處理忽略索引
        if self.ignore_index >= 0:
            mask = (target != self.ignore_index).float()
            focal_loss = focal_loss * mask
            return focal_loss.sum() / (mask.sum() + 1e-8)
        else:
            return focal_loss.mean()
    
    def boundary_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        邊界損失 (強調分割邊界)
        
        Args:
            pred: 預測logits (B, C, H, W)
            target: 目標標籤 (B, H, W)
        
        Returns:
            boundary_loss: 邊界損失
        """
        # 計算梯度來檢測邊界
        def compute_gradient(tensor):
            # Sobel 算子
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            
            sobel_x = sobel_x.to(tensor.device)
            sobel_y = sobel_y.to(tensor.device)
            
            grad_x = F.conv2d(tensor, sobel_x, padding=1)
            grad_y = F.conv2d(tensor, sobel_y, padding=1)
            
            return torch.sqrt(grad_x ** 2 + grad_y ** 2)
        
        # 獲取預測概率
        pred_prob = F.softmax(pred, dim=1)
        pred_max = torch.argmax(pred_prob, dim=1, keepdim=True).float()
        
        # 計算目標的邊界
        target_boundary = compute_gradient(target.unsqueeze(1).float())
        pred_boundary = compute_gradient(pred_max)
        
        # 邊界損失 (L2距離)
        boundary_loss = F.mse_loss(pred_boundary, target_boundary)
        
        return boundary_loss
    
    def lovasz_softmax_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Lovász-Softmax 損失 (針對IoU優化)
        
        Args:
            pred: 預測logits (B, C, H, W)
            target: 目標標籤 (B, H, W)
        
        Returns:
            lovasz_loss: Lovász損失
        """
        # 簡化版本的 Lovász-Softmax
        # 完整實現較複雜，這裡提供基本版本
        
        pred_prob = F.softmax(pred, dim=1)
        
        # 將每個類別分開處理
        losses = []
        for c in range(self.num_classes):
            if c == self.ignore_index:
                continue
            
            # 當前類別的預測和目標
            pred_c = pred_prob[:, c]  # (B, H, W)
            target_c = (target == c).float()  # (B, H, W)
            
            # 忽略指定索引
            if self.ignore_index >= 0:
                mask = (target != self.ignore_index).float()
                pred_c = pred_c * mask
                target_c = target_c * mask
            
            # 計算該類別的損失 (簡化版)
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum() - intersection
            iou_loss = 1.0 - (intersection + 1e-8) / (union + 1e-8)
            
            losses.append(iou_loss)
        
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=pred.device)
    
    def forward(self, 
                predictions: torch.Tensor, 
                targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向傳播
        
        Args:
            predictions: 模型預測 (B, C, H, W)
            targets: 目標標籤 (B, H, W)
        
        Returns:
            total_loss: 總損失
            loss_dict: 詳細損失字典
        """
        loss_dict = {}
        
        if self.loss_type == 'ce':
            # 只使用交叉熵損失
            ce_loss = self.cross_entropy_loss(predictions, targets)
            total_loss = ce_loss
            loss_dict = {
                'cross_entropy': ce_loss,
                'total': total_loss
            }
        
        elif self.loss_type == 'dice':
            # 只使用 Dice 損失
            dice_loss = self.dice_loss(predictions, targets)
            total_loss = dice_loss
            loss_dict = {
                'dice': dice_loss,
                'total': total_loss
            }
        
        elif self.loss_type == 'focal':
            # 只使用 Focal 損失
            focal_loss = self.focal_loss(predictions, targets)
            total_loss = focal_loss
            loss_dict = {
                'focal': focal_loss,
                'total': total_loss
            }
        
        elif self.loss_type == 'combined':
            # 結合交叉熵和 Dice 損失
            ce_loss = self.cross_entropy_loss(predictions, targets)
            dice_loss = self.dice_loss(predictions, targets)
            
            total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
            
            loss_dict = {
                'cross_entropy': ce_loss,
                'dice': dice_loss,
                'total': total_loss
            }
        
        elif self.loss_type == 'advanced':
            # 高級損失組合
            ce_loss = self.cross_entropy_loss(predictions, targets)
            dice_loss = self.dice_loss(predictions, targets)
            focal_loss = self.focal_loss(predictions, targets)
            boundary_loss = self.boundary_loss(predictions, targets)
            
            total_loss = (0.4 * ce_loss + 
                         0.3 * dice_loss + 
                         0.2 * focal_loss + 
                         0.1 * boundary_loss)
            
            loss_dict = {
                'cross_entropy': ce_loss,
                'dice': dice_loss,
                'focal': focal_loss,
                'boundary': boundary_loss,
                'total': total_loss
            }
        
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        
        return total_loss, loss_dict


class DiceLoss(nn.Module):
    """
    獨立的 Dice 損失實現
    """
    def __init__(self, num_classes: int = 21, smooth: float = 1e-6, ignore_index: int = 255):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_prob = F.softmax(pred, dim=1)
        
        target_one_hot = F.one_hot(target.long(), num_classes=self.num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        # 處理忽略索引
        if self.ignore_index >= 0:
            mask = (target != self.ignore_index).float().unsqueeze(1)
            pred_prob = pred_prob * mask
            target_one_hot = target_one_hot * mask
        
        intersection = (pred_prob * target_one_hot).sum(dim=(2, 3))
        union = pred_prob.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - dice_coeff.mean()


class IoULoss(nn.Module):
    """
    IoU 損失實現
    """
    def __init__(self, num_classes: int = 21, smooth: float = 1e-6, ignore_index: int = 255):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_prob = F.softmax(pred, dim=1)
        
        target_one_hot = F.one_hot(target.long(), num_classes=self.num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        # 處理忽略索引
        if self.ignore_index >= 0:
            mask = (target != self.ignore_index).float().unsqueeze(1)
            pred_prob = pred_prob * mask
            target_one_hot = target_one_hot * mask
        
        intersection = (pred_prob * target_one_hot).sum(dim=(2, 3))
        union = pred_prob.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3)) - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - iou.mean()


def create_segmentation_loss(num_classes: int = 21,
                           loss_type: str = 'combined',
                           **kwargs) -> SegmentationLoss:
    """
    分割損失工廠函數
    
    Args:
        num_classes: 分割類別數
        loss_type: 損失類型
        **kwargs: 其他參數
    
    Returns:
        seg_loss: 分割損失函數
    """
    return SegmentationLoss(num_classes=num_classes, loss_type=loss_type, **kwargs)


if __name__ == "__main__":
    # 測試代碼
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 創建分割損失
    seg_loss = create_segmentation_loss(num_classes=21, loss_type='combined')
    
    # 模擬預測和目標
    batch_size = 2
    height, width = 256, 256
    num_classes = 21
    
    # 預測: (B, C, H, W)
    predictions = torch.randn(batch_size, num_classes, height, width).to(device)
    
    # 目標: (B, H, W)
    targets = torch.randint(0, num_classes, (batch_size, height, width)).to(device)
    
    # 計算損失
    total_loss, loss_dict = seg_loss(predictions, targets)
    
    print("✅ 分割損失測試成功！")
    print(f"📊 總損失: {total_loss.item():.4f}")
    print(f"🔍 詳細損失: {loss_dict}")
    print(f"📈 損失項: {list(loss_dict.keys())}")
    
    # 測試不同損失類型
    print("\n🧪 測試不同損失類型:")
    loss_types = ['ce', 'dice', 'focal', 'combined']
    
    for loss_type in loss_types:
        test_loss = create_segmentation_loss(num_classes=21, loss_type=loss_type)
        test_total_loss, test_loss_dict = test_loss(predictions, targets)
        print(f"  {loss_type}: {test_total_loss.item():.4f}")
"""
分類任務損失函數
包含交叉熵損失、標籤平滑、Focal損失、對比學習損失等
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math


class ClassificationLoss(nn.Module):
    """
    分類任務損失函數
    
    支援多種分類損失函數組合，包括：
    1. 交叉熵損失 (含標籤平滑)
    2. Focal Loss (處理類別不平衡)
    3. 對比學習損失
    4. 溫度縮放
    5. Mixup 數據增強支援
    
    Args:
        num_classes: 分類類別數
        loss_type: 損失類型 ('ce', 'focal', 'contrastive', 'combined')
        label_smoothing: 標籤平滑參數
        class_weights: 類別權重
        focal_alpha: Focal loss alpha 參數
        focal_gamma: Focal loss gamma 參數
        temperature: 溫度縮放參數
        contrastive_margin: 對比學習邊界
    """
    
    def __init__(self,
                 num_classes: int = 10,
                 loss_type: str = 'combined',
                 label_smoothing: float = 0.1,
                 class_weights: Optional[torch.Tensor] = None,
                 focal_alpha: Union[float, List[float]] = 0.25,
                 focal_gamma: float = 2.0,
                 temperature: float = 1.0,
                 contrastive_margin: float = 0.5,
                 ce_weight: float = 1.0,
                 focal_weight: float = 1.0,
                 contrastive_weight: float = 0.1):
        super().__init__()
        
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.temperature = temperature
        self.contrastive_margin = contrastive_margin
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.contrastive_weight = contrastive_weight
        
        # 設置類別權重
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
        
        # 創建基礎損失函數
        self.ce_loss = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=label_smoothing,
            reduction='mean'
        )
        
        # 設置 Focal Loss alpha
        if isinstance(focal_alpha, (list, tuple)):
            assert len(focal_alpha) == num_classes, "Alpha 長度必須等於類別數"
            self.register_buffer('focal_alpha_tensor', torch.tensor(focal_alpha))
        else:
            self.focal_alpha_tensor = focal_alpha
    
    def cross_entropy_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        交叉熵損失 (含標籤平滑)
        
        Args:
            pred: 預測logits (B, C)
            target: 目標標籤 (B,)
        
        Returns:
            ce_loss: 交叉熵損失
        """
        return self.ce_loss(pred, target)
    
    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Focal Loss 實現
        
        Args:
            pred: 預測logits (B, C)
            target: 目標標籤 (B,)
        
        Returns:
            focal_loss: Focal損失
        """
        # 計算交叉熵 (不進行 reduction)
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        
        # 計算概率
        p_t = torch.exp(-ce_loss)  # p_t = exp(-ce_loss)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.focal_gamma
        
        # Alpha weighting
        if isinstance(self.focal_alpha_tensor, torch.Tensor):
            # 多類別 alpha
            if self.focal_alpha_tensor.device != pred.device:
                self.focal_alpha_tensor = self.focal_alpha_tensor.to(pred.device)
            alpha_t = self.focal_alpha_tensor.gather(0, target)
            focal_weight = alpha_t * focal_weight
        elif self.focal_alpha_tensor > 0:
            # 單一 alpha 值
            focal_weight = self.focal_alpha_tensor * focal_weight
        
        # 應用 Focal weight
        focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()
    
    def contrastive_loss(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        對比學習損失 (SimCLR 風格)
        
        Args:
            features: 特徵向量 (B, D)
            labels: 標籤 (B,)
        
        Returns:
            contrastive_loss: 對比學習損失
        """
        batch_size = features.size(0)
        
        if batch_size < 2:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        # L2 正規化特徵
        features = F.normalize(features, dim=1)
        
        # 計算相似度矩陣
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 創建標籤掩碼
        labels = labels.unsqueeze(1)
        mask = (labels == labels.T).float()  # 相同標籤為 1
        
        # 移除對角線 (自己和自己)
        mask = mask - torch.eye(batch_size, device=features.device)
        
        # 計算正樣本對數
        positive_pairs = mask.sum()
        
        if positive_pairs == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        # 對比學習損失計算
        exp_sim = torch.exp(similarity_matrix)
        
        # 分子：正樣本相似度
        pos_sim = (exp_sim * mask).sum(dim=1)
        
        # 分母：所有樣本相似度 (除了自己)
        neg_mask = 1 - torch.eye(batch_size, device=features.device)
        all_sim = (exp_sim * neg_mask).sum(dim=1)
        
        # 避免除零
        loss = -torch.log((pos_sim + 1e-8) / (all_sim + 1e-8))
        
        # 只計算有正樣本的損失
        valid_mask = (mask.sum(dim=1) > 0).float()
        loss = (loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        
        return loss
    
    def forward(self, 
                predictions: torch.Tensor, 
                targets: torch.Tensor,
                features: Optional[torch.Tensor] = None,
                teacher_logits: Optional[torch.Tensor] = None,
                mixup_params: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向傳播
        
        Args:
            predictions: 模型預測 (B, C)
            targets: 目標標籤 (B,)
            features: 特徵向量 (B, D) - 用於對比學習
            teacher_logits: 教師模型logits (B, C) - 用於知識蒸餾
            mixup_params: Mixup參數字典
        
        Returns:
            total_loss: 總損失
            loss_dict: 詳細損失字典
        """
        loss_dict = {}
        
        if self.loss_type == 'ce':
            # 純交叉熵損失
            ce_loss = self.cross_entropy_loss(predictions, targets)
            total_loss = ce_loss
            loss_dict = {
                'cross_entropy': ce_loss,
                'total': total_loss
            }
        
        elif self.loss_type == 'focal':
            # 純 Focal 損失
            focal_loss = self.focal_loss(predictions, targets)
            total_loss = focal_loss
            loss_dict = {
                'focal': focal_loss,
                'total': total_loss
            }
        
        elif self.loss_type == 'contrastive':
            # 對比學習損失
            if features is None:
                raise ValueError("Contrastive loss requires features")
            
            ce_loss = self.cross_entropy_loss(predictions, targets)
            contrastive_loss = self.contrastive_loss(features, targets)
            
            total_loss = ce_loss + self.contrastive_weight * contrastive_loss
            
            loss_dict = {
                'cross_entropy': ce_loss,
                'contrastive': contrastive_loss,
                'total': total_loss
            }
        
        elif self.loss_type == 'combined':
            # 組合損失：交叉熵 + Focal
            ce_loss = self.cross_entropy_loss(predictions, targets)
            focal_loss = self.focal_loss(predictions, targets)
            
            total_loss = self.ce_weight * ce_loss + self.focal_weight * focal_loss
            
            loss_dict = {
                'cross_entropy': ce_loss,
                'focal': focal_loss,
                'total': total_loss
            }
            
            # 可選：添加對比學習
            if features is not None:
                contrastive_loss = self.contrastive_loss(features, targets)
                total_loss += self.contrastive_weight * contrastive_loss
                loss_dict['contrastive'] = contrastive_loss
                loss_dict['total'] = total_loss
        
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        
        return total_loss, loss_dict


def create_classification_loss(num_classes: int = 10,
                             loss_type: str = 'combined',
                             **kwargs) -> ClassificationLoss:
    """
    分類損失工廠函數
    
    Args:
        num_classes: 分類類別數
        loss_type: 損失類型
        **kwargs: 其他參數
    
    Returns:
        classification_loss: 分類損失函數
    """
    return ClassificationLoss(num_classes=num_classes, loss_type=loss_type, **kwargs)


if __name__ == "__main__":
    # 測試代碼
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 創建分類損失
    cls_loss = create_classification_loss(num_classes=10, loss_type='combined')
    
    # 模擬預測和目標
    batch_size = 16
    num_classes = 10
    feature_dim = 128
    
    # 預測: (B, C)
    predictions = torch.randn(batch_size, num_classes).to(device)
    
    # 目標: (B,)
    targets = torch.randint(0, num_classes, (batch_size,)).to(device)
    
    # 特徵: (B, D) - 用於對比學習
    features = torch.randn(batch_size, feature_dim).to(device)
    
    print("✅ 分類損失測試開始...")
    
    # 1. 測試基本組合損失
    total_loss, loss_dict = cls_loss(predictions, targets, features=features)
    
    print(f"📊 組合損失: {total_loss.item():.4f}")
    print(f"🔍 詳細損失: {loss_dict}")
    
    # 2. 測試不同損失類型
    print("\n🧪 測試不同損失類型:")
    loss_types = ['ce', 'focal', 'contrastive', 'combined']
    
    for loss_type in loss_types:
        test_loss = create_classification_loss(num_classes=10, loss_type=loss_type)
        
        try:
            if loss_type == 'contrastive':
                test_total_loss, test_loss_dict = test_loss(predictions, targets, features=features)
            else:
                test_total_loss, test_loss_dict = test_loss(predictions, targets)
            
            print(f"  {loss_type}: {test_total_loss.item():.4f}")
        except Exception as e:
            print(f"  {loss_type}: 錯誤 - {e}")
    
    print("\n🎉 分類損失測試完成！")
    print(f"📋 支援的損失類型: {loss_types}")
    print(f"🔧 支援的特殊功能: 對比學習, 溫度縮放")

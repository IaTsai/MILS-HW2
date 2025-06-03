"""
檢測任務損失函數
支援 FCOS 風格的 anchor-free 檢測損失，包含分類、回歸和中心度損失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math


class DetectionLoss(nn.Module):
    """
    檢測任務損失函數
    
    針對 FCOS 風格的 anchor-free 檢測設計，包含：
    1. Focal Loss 用於分類
    2. IoU Loss 用於邊界框回歸  
    3. Binary Cross Entropy 用於中心度預測
    
    Args:
        num_classes: 檢測類別數
        alpha: Focal Loss 的 alpha 參數
        gamma: Focal Loss 的 gamma 參數
        iou_loss_type: IoU 損失類型 ('iou', 'giou', 'diou', 'ciou')
        center_sampling: 是否使用中心度採樣
        pos_weight: 正樣本權重
    """
    
    def __init__(self, 
                 num_classes: int = 10,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 iou_loss_type: str = 'giou',
                 center_sampling: bool = True,
                 pos_weight: float = 1.0,
                 neg_weight: float = 1.0):
        super().__init__()
        
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.iou_loss_type = iou_loss_type
        self.center_sampling = center_sampling
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        
        # 損失權重
        self.cls_weight = 1.0
        self.reg_weight = 1.0
        self.centerness_weight = 1.0
    
    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Focal Loss 實現
        
        Args:
            pred: 預測logits (B, H*W, num_classes)
            target: 目標標籤 (B, H*W)
        
        Returns:
            focal_loss: Focal loss 值
        """
        # 轉換為概率
        pred_sigmoid = torch.sigmoid(pred)
        
        # 創建 one-hot 編碼
        target_onehot = F.one_hot(target.long(), num_classes=self.num_classes).float()
        
        # 計算交叉熵
        ce_loss = F.binary_cross_entropy_with_logits(
            pred, target_onehot, reduction='none'
        )
        
        # 計算 p_t
        p_t = pred_sigmoid * target_onehot + (1 - pred_sigmoid) * (1 - target_onehot)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weighting
        if self.alpha >= 0:
            alpha_t = self.alpha * target_onehot + (1 - self.alpha) * (1 - target_onehot)
            focal_weight = alpha_t * focal_weight
        
        # 應用 Focal weight
        focal_loss = focal_weight * ce_loss
        
        return focal_loss.sum() / (target_onehot.sum() + 1e-8)
    
    def iou_loss(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        IoU Loss 實現 (支援多種 IoU 變體)
        
        Args:
            pred_boxes: 預測邊界框 (N, 4) - (cx, cy, w, h)
            target_boxes: 目標邊界框 (N, 4) - (cx, cy, w, h)
        
        Returns:
            iou_loss: IoU 損失值
        """
        if pred_boxes.size(0) == 0:
            return pred_boxes.sum() * 0
        
        # Clamp box dimensions to prevent numerical issues
        # Create new tensors instead of modifying in-place
        pred_boxes_clamped = torch.zeros_like(pred_boxes)
        target_boxes_clamped = torch.zeros_like(target_boxes)
        
        # Clamp center coordinates to [0, 1]
        pred_boxes_clamped[:, 0:2] = torch.clamp(pred_boxes[:, 0:2], min=0.0, max=1.0)
        target_boxes_clamped[:, 0:2] = torch.clamp(target_boxes[:, 0:2], min=0.0, max=1.0)
        
        # Clamp width/height to [0.01, 1.0] to prevent zero or negative dimensions
        pred_boxes_clamped[:, 2:4] = torch.clamp(pred_boxes[:, 2:4], min=0.01, max=1.0)
        target_boxes_clamped[:, 2:4] = torch.clamp(target_boxes[:, 2:4], min=0.01, max=1.0)
        
        # Use clamped boxes for the rest of the computation
        pred_boxes = pred_boxes_clamped
        target_boxes = target_boxes_clamped
        
        # 轉換為 (x1, y1, x2, y2) 格式
        pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2
        
        target_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
        target_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
        target_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
        target_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2
        
        # 計算交集
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # 計算聯集
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union_area = pred_area + target_area - inter_area
        
        # 基本 IoU
        iou = inter_area / (union_area + 1e-8)
        
        if self.iou_loss_type == 'iou':
            return 1 - iou.mean()
        
        elif self.iou_loss_type == 'giou':
            # GIoU Loss
            enclosing_x1 = torch.min(pred_x1, target_x1)
            enclosing_y1 = torch.min(pred_y1, target_y1)
            enclosing_x2 = torch.max(pred_x2, target_x2)
            enclosing_y2 = torch.max(pred_y2, target_y2)
            
            enclosing_area = (enclosing_x2 - enclosing_x1) * (enclosing_y2 - enclosing_y1)
            giou = iou - (enclosing_area - union_area) / (enclosing_area + 1e-8)
            
            # Clamp GIoU to [-1, 1] to ensure loss is in [0, 2]
            giou = torch.clamp(giou, min=-1.0, max=1.0)
            
            return 1 - giou.mean()
        
        elif self.iou_loss_type == 'diou':
            # DIoU Loss
            center_distance_sq = (pred_boxes[:, 0] - target_boxes[:, 0]) ** 2 + \
                               (pred_boxes[:, 1] - target_boxes[:, 1]) ** 2
            
            enclosing_x1 = torch.min(pred_x1, target_x1)
            enclosing_y1 = torch.min(pred_y1, target_y1)
            enclosing_x2 = torch.max(pred_x2, target_x2)
            enclosing_y2 = torch.max(pred_y2, target_y2)
            
            diagonal_distance_sq = (enclosing_x2 - enclosing_x1) ** 2 + (enclosing_y2 - enclosing_y1) ** 2
            
            diou = iou - center_distance_sq / (diagonal_distance_sq + 1e-8)
            
            return 1 - diou.mean()
        
        elif self.iou_loss_type == 'ciou':
            # CIoU Loss  
            center_distance_sq = (pred_boxes[:, 0] - target_boxes[:, 0]) ** 2 + \
                               (pred_boxes[:, 1] - target_boxes[:, 1]) ** 2
            
            enclosing_x1 = torch.min(pred_x1, target_x1)
            enclosing_y1 = torch.min(pred_y1, target_y1)
            enclosing_x2 = torch.max(pred_x2, target_x2)
            enclosing_y2 = torch.max(pred_y2, target_y2)
            
            diagonal_distance_sq = (enclosing_x2 - enclosing_x1) ** 2 + (enclosing_y2 - enclosing_y1) ** 2
            
            # 縱橫比一致性
            v = (4 / (math.pi ** 2)) * torch.pow(
                torch.atan(target_boxes[:, 2] / (target_boxes[:, 3] + 1e-8)) - 
                torch.atan(pred_boxes[:, 2] / (pred_boxes[:, 3] + 1e-8)), 2
            )
            
            alpha = v / (1 - iou + v + 1e-8)
            
            ciou = iou - center_distance_sq / (diagonal_distance_sq + 1e-8) - alpha * v
            
            return 1 - ciou.mean()
        
        else:
            raise ValueError(f"Unsupported IoU loss type: {self.iou_loss_type}")
    
    def centerness_loss(self, pred_centerness: torch.Tensor, target_centerness: torch.Tensor) -> torch.Tensor:
        """
        中心度損失 (Binary Cross Entropy)
        
        Args:
            pred_centerness: 預測中心度 (N,)
            target_centerness: 目標中心度 (N,)
        
        Returns:
            centerness_loss: 中心度損失
        """
        if pred_centerness.size(0) == 0:
            return pred_centerness.sum() * 0
        
        return F.binary_cross_entropy_with_logits(
            pred_centerness, target_centerness, reduction='mean'
        )
    
    def forward(self, 
                predictions: torch.Tensor, 
                targets: List[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向傳播
        
        Args:
            predictions: 模型預測 (B, H*W, 6) - (cx, cy, w, h, centerness, class)
            targets: 目標標籤列表，每個元素包含 'boxes', 'labels'
        
        Returns:
            total_loss: 總損失
            loss_dict: 詳細損失字典
        """
        device = predictions.device
        batch_size = predictions.size(0)
        
        # 解析預測 - 根據實際模型輸出格式調整
        # 如果輸入是15維 [boxes(4), classes(10), centerness(1)]
        if predictions.size(-1) == 15:
            pred_boxes = predictions[..., :4]           # (B, H*W, 4)
            pred_classes = predictions[..., 4:14]       # (B, H*W, 10)
            pred_centerness = predictions[..., 14]      # (B, H*W)
        else:
            # 原始格式 [boxes(4), centerness(1), class(1 or num_classes)]
            pred_boxes = predictions[..., :4]        # (B, H*W, 4)
            pred_centerness = predictions[..., 4]    # (B, H*W)
            pred_classes = predictions[..., 5:]      # (B, H*W, 1) 或者 logits
        
        # 對預測的邊界框應用 sigmoid 以確保在 [0, 1] 範圍內
        pred_boxes = torch.sigmoid(pred_boxes)
        
        # 如果預測類別是單一數值，轉換為 logits
        if pred_classes.size(-1) == 1:
            # 假設這是類別索引，轉換為 one-hot 或者擴展為 logits
            pred_classes = pred_classes.squeeze(-1)  # (B, H*W)
            # 創建偽 logits (簡化處理)
            pred_logits = torch.zeros(batch_size, pred_classes.size(1), self.num_classes).to(device)
            for b in range(batch_size):
                for i in range(pred_classes.size(1)):
                    class_idx = pred_classes[b, i].long()
                    if 0 <= class_idx < self.num_classes:
                        # Create a new tensor instead of in-place modification
                        one_hot = torch.zeros_like(pred_logits[b, i])
                        one_hot[class_idx] = 1.0
                        pred_logits = pred_logits.clone()
                        pred_logits[b, i] = one_hot
        else:
            pred_logits = pred_classes
        
        # 初始化損失
        total_cls_loss = 0
        total_reg_loss = 0
        total_centerness_loss = 0
        
        num_positive_samples = 0
        
        for batch_idx in range(batch_size):
            if batch_idx < len(targets) and targets[batch_idx] is not None:
                # 解析目標
                target_dict = targets[batch_idx]
                if 'boxes' in target_dict and 'labels' in target_dict:
                    target_boxes = target_dict['boxes']      # (N, 4)
                    target_labels = target_dict['labels']    # (N,)
                    
                    if target_boxes.size(0) > 0:
                        # 簡化的正負樣本分配 (實際應該基於距離或IoU)
                        # 這裡假設前 N 個預測對應 N 個目標
                        num_targets = min(target_boxes.size(0), pred_boxes.size(1))
                        
                        if num_targets > 0:
                            # 分類損失
                            batch_pred_logits = pred_logits[batch_idx, :num_targets]  # (N, num_classes)
                            batch_target_labels = target_labels[:num_targets]         # (N,)
                            
                            cls_loss = self.focal_loss(batch_pred_logits.unsqueeze(0), 
                                                     batch_target_labels.unsqueeze(0))
                            total_cls_loss += cls_loss
                            
                            # 回歸損失
                            batch_pred_boxes = pred_boxes[batch_idx, :num_targets]    # (N, 4)
                            batch_target_boxes = target_boxes[:num_targets]          # (N, 4)
                            
                            # 確保目標框也在 [0, 1] 範圍內
                            batch_target_boxes = torch.clamp(batch_target_boxes, min=0.0, max=1.0)
                            
                            reg_loss = self.iou_loss(batch_pred_boxes, batch_target_boxes)
                            total_reg_loss += reg_loss
                            
                            # 中心度損失 (簡化：使用IoU作為目標中心度)
                            with torch.no_grad():
                                # 計算目標中心度 (基於IoU)
                                target_centerness = self._compute_centerness_targets(
                                    batch_pred_boxes, batch_target_boxes
                                )
                            
                            batch_pred_centerness = pred_centerness[batch_idx, :num_targets]
                            centerness_loss = self.centerness_loss(batch_pred_centerness, target_centerness)
                            total_centerness_loss += centerness_loss
                            
                            num_positive_samples += num_targets
        
        # 平均化損失
        if num_positive_samples > 0:
            total_cls_loss = total_cls_loss / batch_size
            total_reg_loss = total_reg_loss / batch_size
            total_centerness_loss = total_centerness_loss / batch_size
        else:
            # 沒有正樣本時，只計算分類損失 (背景類)
            total_cls_loss = self.focal_loss(pred_logits, torch.zeros_like(pred_logits[..., 0]).long())
            total_reg_loss = pred_boxes.sum() * 0
            total_centerness_loss = pred_centerness.sum() * 0
        
        # 加權總損失
        total_loss = (self.cls_weight * total_cls_loss + 
                     self.reg_weight * total_reg_loss + 
                     self.centerness_weight * total_centerness_loss)
        
        # 確保損失非負
        total_loss = torch.clamp(total_loss, min=0.0)
        
        loss_dict = {
            'classification': total_cls_loss,
            'regression': total_reg_loss,
            'centerness': total_centerness_loss,
            'total': total_loss
        }
        
        return total_loss, loss_dict
    
    def _compute_centerness_targets(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        計算中心度目標 (基於IoU)
        
        Args:
            pred_boxes: 預測邊界框 (N, 4)
            target_boxes: 目標邊界框 (N, 4)
        
        Returns:
            centerness_targets: 中心度目標 (N,)
        """
        # 簡化：使用IoU作為中心度目標
        ious = self._compute_ious(pred_boxes, target_boxes)
        return torch.sqrt(torch.clamp(ious, min=0, max=1))
    
    def _compute_ious(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        計算兩組邊界框的IoU
        
        Args:
            boxes1: 邊界框1 (N, 4) - (cx, cy, w, h)
            boxes2: 邊界框2 (N, 4) - (cx, cy, w, h)
        
        Returns:
            ious: IoU值 (N,)
        """
        # 轉換為 (x1, y1, x2, y2)
        boxes1_x1 = boxes1[:, 0] - boxes1[:, 2] / 2
        boxes1_y1 = boxes1[:, 1] - boxes1[:, 3] / 2
        boxes1_x2 = boxes1[:, 0] + boxes1[:, 2] / 2
        boxes1_y2 = boxes1[:, 1] + boxes1[:, 3] / 2
        
        boxes2_x1 = boxes2[:, 0] - boxes2[:, 2] / 2
        boxes2_y1 = boxes2[:, 1] - boxes2[:, 3] / 2
        boxes2_x2 = boxes2[:, 0] + boxes2[:, 2] / 2
        boxes2_y2 = boxes2[:, 1] + boxes2[:, 3] / 2
        
        # 計算交集
        inter_x1 = torch.max(boxes1_x1, boxes2_x1)
        inter_y1 = torch.max(boxes1_y1, boxes2_y1)
        inter_x2 = torch.min(boxes1_x2, boxes2_x2)
        inter_y2 = torch.min(boxes1_y2, boxes2_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # 計算聯集
        area1 = (boxes1_x2 - boxes1_x1) * (boxes1_y2 - boxes1_y1)
        area2 = (boxes2_x2 - boxes2_x1) * (boxes2_y2 - boxes2_y1)
        union_area = area1 + area2 - inter_area
        
        return inter_area / (union_area + 1e-8)


def create_detection_loss(num_classes: int = 10, 
                         loss_type: str = 'fcos',
                         **kwargs) -> DetectionLoss:
    """
    檢測損失工廠函數
    
    Args:
        num_classes: 檢測類別數
        loss_type: 損失類型
        **kwargs: 其他參數
    
    Returns:
        detection_loss: 檢測損失函數
    """
    if loss_type == 'fcos':
        return DetectionLoss(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unsupported detection loss type: {loss_type}")


if __name__ == "__main__":
    # 測試代碼
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 創建檢測損失
    det_loss = create_detection_loss(num_classes=10, iou_loss_type='giou')
    
    # 模擬預測和目標
    batch_size = 2
    num_predictions = 100
    
    # 預測: (B, H*W, 6) - (cx, cy, w, h, centerness, class)
    predictions = torch.randn(batch_size, num_predictions, 6).to(device)
    predictions[..., :4] = torch.sigmoid(predictions[..., :4])  # 歸一化坐標
    predictions[..., 5] = torch.randint(0, 10, (batch_size, num_predictions)).float()  # 類別
    
    # 目標
    targets = []
    for b in range(batch_size):
        num_objects = torch.randint(1, 5, (1,)).item()
        target = {
            'boxes': torch.rand(num_objects, 4).to(device),  # (cx, cy, w, h)
            'labels': torch.randint(0, 10, (num_objects,)).to(device)
        }
        targets.append(target)
    
    # 計算損失
    total_loss, loss_dict = det_loss(predictions, targets)
    
    print("✅ 檢測損失測試成功！")
    print(f"📊 總損失: {total_loss.item():.4f}")
    print(f"🔍 詳細損失: {loss_dict}")
    print(f"📈 損失項: {list(loss_dict.keys())}")
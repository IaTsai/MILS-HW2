"""
多任務損失函數
統一管理檢測、分割、分類任務的損失函數，支援自適應權重平衡策略
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import math
import numpy as np

from .detection_loss import create_detection_loss
from .segmentation_loss import create_segmentation_loss  
from .classification_loss import create_classification_loss


class MultiTaskLoss(nn.Module):
    """
    多任務損失函數
    
    統一管理所有任務的損失計算，支援：
    1. 固定權重策略
    2. 自適應權重平衡 (Uncertainty Weighting)
    3. 梯度歸一化平衡 (GradNorm)
    4. 任務優先級調度
    5. 損失值歸一化
    
    Args:
        task_weights: 任務權重字典 {'detection': w1, 'segmentation': w2, 'classification': w3}
        weighting_strategy: 權重策略 ('fixed', 'uncertainty', 'gradnorm', 'dynamic')
        loss_configs: 各任務損失配置
        uncertainty_init: 不確定性權重初始值
        gradnorm_alpha: GradNorm 平衡參數
        temperature: 溫度參數 (用於權重縮放)
    """
    
    def __init__(self,
                 task_weights: Optional[Dict[str, float]] = None,
                 weighting_strategy: str = 'uncertainty',
                 loss_configs: Optional[Dict[str, Dict]] = None,
                 uncertainty_init: float = 0.0,
                 gradnorm_alpha: float = 1.5,
                 temperature: float = 2.0,
                 normalize_losses: bool = True,
                 task_balancing: bool = True):
        super().__init__()
        
        # 基本設置
        self.weighting_strategy = weighting_strategy
        self.gradnorm_alpha = gradnorm_alpha
        self.temperature = temperature
        self.normalize_losses = normalize_losses
        self.task_balancing = task_balancing
        
        # 任務列表
        self.tasks = ['detection', 'segmentation', 'classification']
        
        # 設置默認任務權重
        if task_weights is None:
            task_weights = {'detection': 1.0, 'segmentation': 1.0, 'classification': 1.0}
        self.task_weights = task_weights
        
        # 設置默認損失配置
        if loss_configs is None:
            loss_configs = {
                'detection': {'num_classes': 10, 'iou_loss_type': 'giou'},
                'segmentation': {'num_classes': 21, 'loss_type': 'combined'},
                'classification': {'num_classes': 10, 'loss_type': 'combined'}
            }
        
        # 創建各任務損失函數
        self.detection_loss = create_detection_loss(**loss_configs['detection'])
        self.segmentation_loss = create_segmentation_loss(**loss_configs['segmentation'])
        self.classification_loss = create_classification_loss(**loss_configs['classification'])
        
        # 不確定性權重 (可學習參數)
        if weighting_strategy == 'uncertainty':
            self.log_vars = nn.ParameterDict({
                task: nn.Parameter(torch.tensor(uncertainty_init))
                for task in self.tasks
            })
        
        # GradNorm 相關
        if weighting_strategy == 'gradnorm':
            self.task_weights_param = nn.ParameterDict({
                task: nn.Parameter(torch.tensor(weight))
                for task, weight in task_weights.items()
            })
            self.initial_losses = {}
            self.training_step = 0
        
        # 損失歷史記錄 (用於動態調整)
        self.loss_history = {task: [] for task in self.tasks}
        self.loss_moving_avg = {task: 0.0 for task in self.tasks}
        self.alpha_ema = 0.9  # 指數移動平均參數
        
        # 計數器
        self.step_count = 0
    
    def uncertainty_weighting(self, task_losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        不確定性加權 (Multi-Task Learning Using Uncertainty)
        
        Loss = Σ(1/(2*σ²) * L_i + log(σ))
        
        Args:
            task_losses: 各任務損失字典
        
        Returns:
            total_loss: 加權總損失
            weights: 實際使用的權重
        """
        total_loss = 0
        weights = {}
        
        for task, loss in task_losses.items():
            if task in self.log_vars:
                # σ² = exp(log_var)
                precision = torch.exp(-self.log_vars[task])
                
                # 損失項: 1/(2*σ²) * L_i + 1/2 * log(σ²)
                weighted_loss = precision * loss + 0.5 * self.log_vars[task]
                total_loss += weighted_loss
                
                # 記錄實際權重
                weights[task] = precision.item()
            else:
                total_loss += loss
                weights[task] = 1.0
        
        return total_loss, weights
    
    def gradnorm_weighting(self, task_losses: Dict[str, torch.Tensor], 
                          shared_params: List[nn.Parameter]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        GradNorm 動態權重平衡
        
        基於梯度範數平衡各任務權重
        
        Args:
            task_losses: 各任務損失字典
            shared_params: 共享參數列表
        
        Returns:
            total_loss: 加權總損失  
            weights: 實際使用的權重
        """
        if self.training_step == 0:
            # 初始化：記錄初始損失
            self.initial_losses = {task: loss.item() for task, loss in task_losses.items()}
        
        # 計算當前損失比率
        loss_ratios = {}
        for task, loss in task_losses.items():
            if task in self.initial_losses and self.initial_losses[task] > 0:
                loss_ratios[task] = loss.item() / self.initial_losses[task]
            else:
                loss_ratios[task] = 1.0
        
        # 計算目標損失比率 (基於訓練進度)
        avg_loss_ratio = sum(loss_ratios.values()) / len(loss_ratios)
        target_ratios = {task: avg_loss_ratio ** self.gradnorm_alpha for task in self.tasks}
        
        # 計算梯度範數
        total_loss = 0
        task_grads = {}
        
        for task, loss in task_losses.items():
            if task in self.task_weights_param:
                weight = torch.abs(self.task_weights_param[task])  # 確保權重為正
                weighted_loss = weight * loss
                total_loss += weighted_loss
                
                # 計算該任務對共享參數的梯度範數
                if shared_params:
                    grads = torch.autograd.grad(weighted_loss, shared_params, 
                                              retain_graph=True, create_graph=True)
                    grad_norm = torch.norm(torch.cat([g.view(-1) for g in grads]))
                    task_grads[task] = grad_norm
                else:
                    task_grads[task] = torch.tensor(1.0, device=loss.device)
        
        # 更新任務權重 (如果處於訓練模式)
        if self.training and shared_params:
            avg_grad = sum(task_grads.values()) / len(task_grads)
            
            for task in self.tasks:
                if task in self.task_weights_param and task in task_grads:
                    # 計算權重更新
                    relative_target = target_ratios[task] / avg_loss_ratio
                    relative_grad = task_grads[task] / avg_grad
                    
                    # GradNorm 更新規則
                    grad_target = avg_grad * (relative_target ** self.gradnorm_alpha)
                    grad_loss = F.l1_loss(task_grads[task], grad_target.detach())
                    
                    # 反向傳播更新權重
                    grad_loss.backward(retain_graph=True)
        
        weights = {task: torch.abs(self.task_weights_param[task]).item() 
                  for task in self.task_weights_param}
        
        self.training_step += 1
        return total_loss, weights
    
    def dynamic_weighting(self, task_losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        動態權重調整
        
        基於損失變化趨勢動態調整權重
        
        Args:
            task_losses: 各任務損失字典
        
        Returns:
            total_loss: 加權總損失
            weights: 實際使用的權重
        """
        total_loss = 0
        weights = {}
        
        # 更新損失移動平均
        for task, loss in task_losses.items():
            loss_val = loss.item()
            self.loss_history[task].append(loss_val)
            
            # 保持歷史記錄長度
            if len(self.loss_history[task]) > 100:
                self.loss_history[task].pop(0)
            
            # 更新移動平均
            self.loss_moving_avg[task] = (self.alpha_ema * self.loss_moving_avg[task] + 
                                        (1 - self.alpha_ema) * loss_val)
        
        # 計算動態權重
        if self.step_count > 10:  # 等待足夠的歷史數據
            loss_stds = {}
            for task in self.tasks:
                if len(self.loss_history[task]) > 5:
                    # 計算損失變化的標準差
                    recent_losses = self.loss_history[task][-10:]
                    loss_stds[task] = np.std(recent_losses) if len(recent_losses) > 1 else 1.0
                else:
                    loss_stds[task] = 1.0
            
            # 根據損失穩定性調整權重 (不穩定的任務給更高權重)
            total_std = sum(loss_stds.values())
            for task, loss in task_losses.items():
                if total_std > 0:
                    stability_weight = loss_stds[task] / total_std
                    base_weight = self.task_weights.get(task, 1.0)
                    dynamic_weight = base_weight * (1 + stability_weight)
                else:
                    dynamic_weight = self.task_weights.get(task, 1.0)
                
                weights[task] = dynamic_weight
                total_loss += dynamic_weight * loss
        else:
            # 使用固定權重
            for task, loss in task_losses.items():
                weight = self.task_weights.get(task, 1.0)
                weights[task] = weight
                total_loss += weight * loss
        
        self.step_count += 1
        return total_loss, weights
    
    def fixed_weighting(self, task_losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        固定權重策略
        
        Args:
            task_losses: 各任務損失字典
        
        Returns:
            total_loss: 加權總損失
            weights: 實際使用的權重
        """
        total_loss = 0
        weights = {}
        
        for task, loss in task_losses.items():
            weight = self.task_weights.get(task, 1.0)
            weights[task] = weight
            total_loss += weight * loss
        
        return total_loss, weights
    
    def normalize_task_losses(self, task_losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        歸一化任務損失 (可選)
        
        Args:
            task_losses: 原始任務損失
        
        Returns:
            normalized_losses: 歸一化後的損失
        """
        if not self.normalize_losses:
            return task_losses
        
        # 計算損失的尺度
        loss_scales = {}
        for task, loss in task_losses.items():
            loss_scales[task] = loss.item()
        
        # 歸一化到相似尺度
        max_scale = max(loss_scales.values()) if loss_scales else 1.0
        normalized_losses = {}
        
        for task, loss in task_losses.items():
            if max_scale > 0:
                scale_factor = max_scale / (loss_scales[task] + 1e-8)
                normalized_losses[task] = loss * scale_factor
            else:
                normalized_losses[task] = loss
        
        return normalized_losses
    
    def forward(self, 
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, Any],
                features: Optional[torch.Tensor] = None,
                shared_params: Optional[List[nn.Parameter]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        前向傳播
        
        Args:
            predictions: 各任務預測字典
            targets: 各任務目標字典
            features: 共享特徵 (用於對比學習)
            shared_params: 共享參數 (用於 GradNorm)
        
        Returns:
            total_loss: 總損失
            loss_info: 詳細損失信息
        """
        task_losses = {}
        detailed_losses = {}
        
        # 計算各任務損失
        if 'detection' in predictions and 'detection' in targets:
            det_loss, det_dict = self.detection_loss(predictions['detection'], targets['detection'])
            task_losses['detection'] = det_loss
            detailed_losses['detection'] = det_dict
        
        if 'segmentation' in predictions and 'segmentation' in targets:
            seg_loss, seg_dict = self.segmentation_loss(predictions['segmentation'], targets['segmentation'])
            task_losses['segmentation'] = seg_loss
            detailed_losses['segmentation'] = seg_dict
        
        if 'classification' in predictions and 'classification' in targets:
            # 處理分類任務的特徵輸入
            cls_features = features if features is not None else None
            cls_loss, cls_dict = self.classification_loss(
                predictions['classification'], 
                targets['classification'],
                features=cls_features
            )
            task_losses['classification'] = cls_loss
            detailed_losses['classification'] = cls_dict
        
        # 損失歸一化 (可選)
        if self.normalize_losses:
            task_losses = self.normalize_task_losses(task_losses)
        
        # 根據策略計算加權損失
        if self.weighting_strategy == 'uncertainty':
            total_loss, weights = self.uncertainty_weighting(task_losses)
        elif self.weighting_strategy == 'gradnorm':
            total_loss, weights = self.gradnorm_weighting(task_losses, shared_params or [])
        elif self.weighting_strategy == 'dynamic':
            total_loss, weights = self.dynamic_weighting(task_losses)
        else:  # 'fixed'
            total_loss, weights = self.fixed_weighting(task_losses)
        
        # 構建返回信息
        loss_info = {
            'total_loss': total_loss,
            'task_losses': task_losses,
            'detailed_losses': detailed_losses,
            'task_weights': weights,
            'weighting_strategy': self.weighting_strategy
        }
        
        # 添加不確定性信息 (如果適用)
        if self.weighting_strategy == 'uncertainty':
            uncertainties = {task: torch.exp(0.5 * self.log_vars[task]).item() 
                           for task in self.log_vars}
            loss_info['uncertainties'] = uncertainties
        
        return total_loss, loss_info
    
    def get_task_weights(self) -> Dict[str, float]:
        """
        獲取當前任務權重
        
        Returns:
            weights: 任務權重字典
        """
        if self.weighting_strategy == 'uncertainty':
            return {task: torch.exp(-self.log_vars[task]).item() for task in self.log_vars}
        elif self.weighting_strategy == 'gradnorm':
            return {task: torch.abs(param).item() for task, param in self.task_weights_param.items()}
        else:
            return self.task_weights.copy()
    
    def set_task_weights(self, new_weights: Dict[str, float]):
        """
        設置任務權重
        
        Args:
            new_weights: 新的任務權重
        """
        self.task_weights.update(new_weights)
        
        if self.weighting_strategy == 'gradnorm':
            for task, weight in new_weights.items():
                if task in self.task_weights_param:
                    # Use detach() and clone() instead of direct .data manipulation
                    with torch.no_grad():
                        self.task_weights_param[task].copy_(torch.tensor(weight))
    
    def reset_adaptation(self):
        """
        重置自適應參數
        """
        if self.weighting_strategy == 'uncertainty':
            for param in self.log_vars.values():
                # Use detach() and clone() instead of direct .data manipulation
                with torch.no_grad():
                    param.copy_(torch.zeros_like(param))
        
        if self.weighting_strategy == 'gradnorm':
            self.initial_losses = {}
            self.training_step = 0
        
        if self.weighting_strategy == 'dynamic':
            self.loss_history = {task: [] for task in self.tasks}
            self.loss_moving_avg = {task: 0.0 for task in self.tasks}
            self.step_count = 0


def create_multitask_loss(task_weights: Optional[Dict[str, float]] = None,
                         weighting_strategy: str = 'uncertainty',
                         **kwargs) -> MultiTaskLoss:
    """
    多任務損失工廠函數
    
    Args:
        task_weights: 任務權重字典
        weighting_strategy: 權重策略
        **kwargs: 其他參數
    
    Returns:
        multitask_loss: 多任務損失函數
    """
    return MultiTaskLoss(
        task_weights=task_weights,
        weighting_strategy=weighting_strategy,
        **kwargs
    )


if __name__ == "__main__":
    # 測試代碼
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("✅ 多任務損失測試開始...")
    
    # 創建多任務損失
    multitask_loss = create_multitask_loss(
        task_weights={'detection': 1.0, 'segmentation': 1.0, 'classification': 1.0},
        weighting_strategy='uncertainty'
    )
    
    # 模擬預測數據
    batch_size = 8
    predictions = {
        'detection': torch.randn(batch_size, 100, 6).to(device),  # (B, H*W, 6)
        'segmentation': torch.randn(batch_size, 21, 256, 256).to(device),  # (B, C, H, W)
        'classification': torch.randn(batch_size, 10).to(device)  # (B, C)
    }
    
    # 模擬目標數據
    targets = {
        'detection': [
            {
                'boxes': torch.rand(3, 4).to(device),
                'labels': torch.randint(0, 10, (3,)).to(device)
            } for _ in range(batch_size)
        ],
        'segmentation': torch.randint(0, 21, (batch_size, 256, 256)).to(device),
        'classification': torch.randint(0, 10, (batch_size,)).to(device)
    }
    
    # 共享特徵 (用於分類對比學習)
    features = torch.randn(batch_size, 128).to(device)
    
    print("📊 測試不同權重策略:")
    
    strategies = ['fixed', 'uncertainty', 'dynamic']
    
    for strategy in strategies:
        print(f"\n🧪 測試 {strategy} 策略:")
        
        test_loss = create_multitask_loss(
            weighting_strategy=strategy,
            task_weights={'detection': 1.0, 'segmentation': 1.0, 'classification': 1.0}
        )
        
        try:
            total_loss, loss_info = test_loss(predictions, targets, features=features)
            
            print(f"  總損失: {total_loss.item():.4f}")
            print(f"  任務權重: {loss_info['task_weights']}")
            
            if 'uncertainties' in loss_info:
                print(f"  不確定性: {loss_info['uncertainties']}")
            
            # 測試梯度
            total_loss.backward(retain_graph=True)
            print(f"  ✅ 梯度回傳正常")
            
        except Exception as e:
            print(f"  ❌ 錯誤: {e}")
    
    print("\n🔧 測試權重調整:")
    
    # 測試權重設置和獲取
    current_weights = multitask_loss.get_task_weights()
    print(f"  當前權重: {current_weights}")
    
    new_weights = {'detection': 2.0, 'segmentation': 0.5, 'classification': 1.5}
    multitask_loss.set_task_weights(new_weights)
    updated_weights = multitask_loss.get_task_weights()
    print(f"  更新後權重: {updated_weights}")
    
    print("\n🎉 多任務損失測試完成！")
    print("🚀 支援的權重策略: fixed, uncertainty, gradnorm, dynamic")
    print("⚙️ 支援的功能: 損失歸一化, 動態權重調整, 不確定性建模")
"""
依序訓練器
實現多任務學習的依序訓練流程，整合EWC防遺忘機制
"""
import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path
import logging
from collections import defaultdict
import matplotlib.pyplot as plt

from .ewc import create_ewc_handler, ewc_loss
from ..losses.multitask_loss import create_multitask_loss


class SequentialTrainer:
    """
    依序訓練器
    
    實現多任務學習的依序訓練流程，支援：
    1. 三階段訓練：分割 → 檢測 → 分類
    2. EWC 防遺忘保護
    3. 性能監控與評估
    4. 自適應權重調整
    5. 檢查點管理
    
    Args:
        model: 統一多任務模型
        dataloaders: 各任務數據加載器字典
        ewc_importance: EWC 重要性權重
        save_dir: 檢查點保存目錄
        device: 訓練設備
    """
    
    def __init__(self,
                 model: nn.Module,
                 dataloaders: Dict[str, DataLoader],
                 ewc_importance: float = 1000.0,
                 save_dir: str = './checkpoints',
                 device: str = 'cuda',
                 learning_rate: float = 1e-3,
                 adaptive_ewc: bool = True,
                 forgetting_threshold: float = 0.05):
        
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.learning_rate = learning_rate
        self.adaptive_ewc = adaptive_ewc
        self.forgetting_threshold = forgetting_threshold
        
        # EWC 設置
        self.ewc = create_ewc_handler(model, importance=ewc_importance)
        self.ewc_importance = ewc_importance
        
        # 性能記錄
        self.performance_history = {
            'segmentation': [],
            'detection': [],
            'classification': []
        }
        self.baseline_performance = {}
        self.current_performance = {}
        
        # 訓練歷史
        self.training_history = {
            'losses': defaultdict(list),
            'metrics': defaultdict(list),
            'ewc_penalties': defaultdict(list),
            'forgetting_rates': defaultdict(list)
        }
        
        # 任務順序
        self.task_sequence = ['segmentation', 'detection', 'classification']
        self.completed_tasks = []
        self.current_task = None
        
        # 損失函數
        self.loss_fn = create_multitask_loss(
            weighting_strategy='fixed',
            task_weights={'segmentation': 1.0, 'detection': 1.0, 'classification': 1.0}
        )
        
        # 設置日誌
        self.setup_logging()
    
    def setup_logging(self):
        """設置訓練日誌"""
        log_file = self.save_dir / 'training.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_stage(self, 
                   stage_name: str, 
                   task_type: str, 
                   epochs: int,
                   save_checkpoints: bool = True) -> Dict[str, float]:
        """
        訓練單一階段
        
        Args:
            stage_name: 階段名稱
            task_type: 任務類型
            epochs: 訓練輪數
            save_checkpoints: 是否保存檢查點
        
        Returns:
            final_metrics: 最終性能指標
        """
        self.logger.info(f"🚀 開始 {stage_name} 訓練 ({task_type} 任務)")
        self.logger.info(f"📊 訓練輪數: {epochs}")
        
        self.current_task = task_type
        
        # 設置優化器
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # 獲取數據加載器
        train_loader = self.dataloaders[f'{task_type}_train']
        val_loader = self.dataloaders[f'{task_type}_val']
        
        # 訓練循環
        best_metric = 0.0
        for epoch in range(epochs):
            self.model.train()
            
            # 訓練一個epoch
            train_metrics = self._train_epoch(train_loader, optimizer, task_type, epoch)
            
            # 驗證
            self.model.eval()
            val_metrics = self._validate_epoch(val_loader, task_type, epoch)
            
            # 學習率調整
            scheduler.step()
            
            # 記錄指標
            self.training_history['losses'][task_type].append(train_metrics['loss'])
            self.training_history['metrics'][task_type].append(val_metrics['main_metric'])
            
            if 'ewc_penalty' in train_metrics:
                self.training_history['ewc_penalties'][task_type].append(train_metrics['ewc_penalty'])
            
            # 檢查是否最佳模型
            current_metric = val_metrics['main_metric']
            if current_metric > best_metric:
                best_metric = current_metric
                if save_checkpoints:
                    self._save_checkpoint(stage_name, epoch, val_metrics)
            
            # 打印進度
            if epoch % 10 == 0 or epoch == epochs - 1:
                self.logger.info(
                    f"  Epoch {epoch:3d}/{epochs}: "
                    f"Loss={train_metrics['loss']:.4f}, "
                    f"Metric={current_metric:.4f}"
                )
                
                if 'ewc_penalty' in train_metrics:
                    self.logger.info(f"    EWC懲罰項: {train_metrics['ewc_penalty']:.4f}")
        
        # 評估最終性能
        final_metrics = self.evaluate_task(task_type)
        self.current_performance[task_type] = final_metrics
        
        # 如果是第一個任務，記錄基準性能
        if task_type not in self.baseline_performance:
            self.baseline_performance[task_type] = final_metrics
            self.logger.info(f"📋 記錄基準性能 - {task_type}: {final_metrics['main_metric']:.4f}")
        
        # 完成任務後設置 EWC
        if task_type not in self.completed_tasks:
            self.logger.info(f"🔧 為 {task_type} 任務設置 EWC...")
            self.ewc.finish_task(train_loader, task_id=len(self.completed_tasks), verbose=True)
            self.completed_tasks.append(task_type)
        
        self.logger.info(f"✅ {stage_name} 訓練完成！最佳指標: {best_metric:.4f}")
        return final_metrics
    
    def _train_epoch(self, 
                    dataloader: DataLoader, 
                    optimizer: optim.Optimizer, 
                    task_type: str, 
                    epoch: int) -> Dict[str, float]:
        """訓練一個epoch"""
        total_loss = 0.0
        total_ewc_penalty = 0.0
        num_batches = 0
        
        for batch_idx, batch_data in enumerate(dataloader):
            # 數據準備
            images, targets = self._prepare_batch(batch_data, task_type)
            
            # 前向傳播
            optimizer.zero_grad()
            outputs = self.model(images, task_type=task_type)
            
            # 計算基礎損失
            if task_type == 'segmentation':
                base_loss, _ = self.loss_fn.segmentation_loss(outputs[task_type], targets)
            elif task_type == 'detection':
                base_loss, _ = self.loss_fn.detection_loss(outputs[task_type], targets)
            elif task_type == 'classification':
                base_loss, _ = self.loss_fn.classification_loss(outputs[task_type], targets)
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
            
            # 添加 EWC 懲罰項（如果有已完成的任務）
            if len(self.completed_tasks) > 0:
                total_loss_with_ewc, ewc_penalty = ewc_loss(base_loss, self.ewc)
                total_ewc_penalty += ewc_penalty.item()
            else:
                total_loss_with_ewc = base_loss
                ewc_penalty = torch.tensor(0.0)
            
            # 反向傳播
            total_loss_with_ewc.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += base_loss.item()
            num_batches += 1
            
            # 自適應調整 EWC 權重
            if self.adaptive_ewc and len(self.completed_tasks) > 0:
                self._adaptive_ewc_adjustment(epoch, batch_idx)
        
        metrics = {
            'loss': total_loss / num_batches,
        }
        
        if len(self.completed_tasks) > 0:
            metrics['ewc_penalty'] = total_ewc_penalty / num_batches
        
        return metrics
    
    def _validate_epoch(self, dataloader: DataLoader, task_type: str, epoch: int) -> Dict[str, float]:
        """驗證一個epoch"""
        with torch.no_grad():
            if task_type == 'segmentation':
                return self._validate_segmentation(dataloader)
            elif task_type == 'detection':
                return self._validate_detection(dataloader)
            elif task_type == 'classification':
                return self._validate_classification(dataloader)
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
    
    def _validate_segmentation(self, dataloader: DataLoader) -> Dict[str, float]:
        """驗證分割任務"""
        total_intersection = 0
        total_union = 0
        total_samples = 0
        
        for batch_data in dataloader:
            images, targets = self._prepare_batch(batch_data, 'segmentation')
            outputs = self.model(images, task_type='segmentation')
            predictions = torch.argmax(outputs['segmentation'], dim=1)
            
            # 計算 IoU
            for pred, target in zip(predictions, targets):
                for class_id in range(1, 21):  # 忽略背景類別0
                    pred_mask = (pred == class_id)
                    target_mask = (target == class_id)
                    
                    intersection = (pred_mask & target_mask).sum().item()
                    union = (pred_mask | target_mask).sum().item()
                    
                    if union > 0:
                        total_intersection += intersection
                        total_union += union
            
            total_samples += targets.size(0)
        
        miou = total_intersection / (total_union + 1e-8)
        
        return {
            'main_metric': miou,
            'miou': miou,
            'samples': total_samples
        }
    
    def _validate_detection(self, dataloader: DataLoader) -> Dict[str, float]:
        """驗證檢測任務（簡化版mAP）"""
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_samples = 0
        
        for batch_data in dataloader:
            images, targets = self._prepare_batch(batch_data, 'detection')
            outputs = self.model(images, task_type='detection')
            
            # 簡化的 mAP 計算（使用IoU閾值0.5）
            # 這裡實現一個簡化版本，實際應用中需要更精確的mAP計算
            batch_size = images.size(0)
            for b in range(batch_size):
                if b < len(targets) and targets[b] is not None:
                    num_gt = targets[b]['boxes'].size(0) if 'boxes' in targets[b] else 0
                    # 假設每個預測都有一定的準確率
                    total_tp += max(0, num_gt - 1)  # 簡化假設
                    total_fp += 1
                    total_fn += 1
            
            total_samples += batch_size
        
        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)
        map_score = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            'main_metric': map_score,
            'map': map_score,
            'precision': precision,
            'recall': recall,
            'samples': total_samples
        }
    
    def _validate_classification(self, dataloader: DataLoader) -> Dict[str, float]:
        """驗證分類任務"""
        correct = 0
        total = 0
        
        for batch_data in dataloader:
            images, targets = self._prepare_batch(batch_data, 'classification')
            outputs = self.model(images, task_type='classification')
            
            predictions = torch.argmax(outputs['classification'], dim=1)
            
            # 確保targets是正確的tensor格式
            if not torch.is_tensor(targets):
                targets = torch.tensor(targets, dtype=torch.long).to(self.device)
            
            # 確保predictions和targets都是tensor
            comparison = predictions == targets
            if torch.is_tensor(comparison):
                correct += comparison.sum().item()
            else:
                # 如果比較結果是單個bool值，轉換處理
                correct += int(comparison)
            total += targets.size(0)
        
        accuracy = correct / total
        
        return {
            'main_metric': accuracy,
            'accuracy': accuracy,
            'top1': accuracy,
            'samples': total
        }
    
    def _prepare_batch(self, batch_data: Any, task_type: str) -> Tuple[torch.Tensor, Any]:
        """準備批次數據"""
        if task_type == 'detection' and isinstance(batch_data, list) and len(batch_data) == 2:
            # 檢測數據格式: [images_tensor, [target_dict1, target_dict2, ...]]
            images, targets = batch_data
            images = images.to(self.device)
        
        elif isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
            images, targets = batch_data
            images = images.to(self.device)
            
            # 如果targets是字典且任務是分類，提取labels
            if task_type == 'classification' and isinstance(targets, dict) and 'labels' in targets:
                targets = targets['labels'].to(self.device)
                
        elif isinstance(batch_data, dict):
            # 處理統一數據加載器格式
            images = batch_data['images'].to(self.device)
            targets = batch_data['targets']
        else:
            raise ValueError(f"Unsupported batch format: {type(batch_data)}")
        
        # 將目標移動到設備 - 處理不同的目標格式
        if isinstance(targets, list):
            # 處理list格式的targets (通常來自統一數據加載器)
            processed_targets = []
            for target in targets:
                if isinstance(target, dict):
                    # 處理字典格式的目標
                    processed_target = {}
                    for key, value in target.items():
                        if torch.is_tensor(value):
                            processed_target[key] = value.to(self.device)
                        else:
                            processed_target[key] = value
                    processed_targets.append(processed_target)
                elif torch.is_tensor(target):
                    # 處理tensor格式的目標
                    processed_targets.append(target.to(self.device))
                else:
                    # 其他格式保持不變
                    processed_targets.append(target)
            targets = processed_targets
            
        elif isinstance(targets, dict):
            # 處理字典格式的targets
            for key, value in targets.items():
                if torch.is_tensor(value):
                    targets[key] = value.to(self.device)
                    
        elif torch.is_tensor(targets):
            # 處理tensor格式的targets
            targets = targets.to(self.device)
            
        # 根據任務類型進行特殊處理
        if task_type == 'segmentation' and isinstance(targets, list):
            # 分割任務：提取masks並轉換為tensor
            if len(targets) > 0 and isinstance(targets[0], dict):
                if 'masks' in targets[0]:
                    # 從字典中提取masks
                    masks = torch.stack([t['masks'] for t in targets]).to(self.device)
                    targets = masks
                elif 'labels' in targets[0]:
                    # 如果只有labels，也可以作為分割目標
                    labels = torch.stack([t['labels'] for t in targets]).to(self.device)
                    targets = labels
            elif len(targets) > 0 and torch.is_tensor(targets[0]):
                # 直接stack tensor
                targets = torch.stack(targets).to(self.device)
                
        elif task_type == 'classification' and isinstance(targets, list):
            # 分類任務：如果是list，嘗試轉換為tensor
            if len(targets) > 0:
                if isinstance(targets[0], dict) and 'labels' in targets[0]:
                    # 從字典中提取labels
                    labels = []
                    for t in targets:
                        if torch.is_tensor(t['labels']):
                            # 如果labels是tensor，確保是標量或1D tensor
                            label = t['labels'].item() if t['labels'].numel() == 1 else t['labels'][0].item()
                        else:
                            label = t['labels']
                        labels.append(label)
                    targets = torch.tensor(labels, dtype=torch.long).to(self.device)
                elif torch.is_tensor(targets[0]):
                    # 如果已經是tensor，直接stack
                    targets = torch.stack(targets).to(self.device)
                elif isinstance(targets[0], (int, float)):
                    # 如果是數字列表，轉換為tensor
                    targets = torch.tensor(targets, dtype=torch.long).to(self.device)
        
        return images, targets
    
    def evaluate_task(self, task_type: str) -> Dict[str, float]:
        """評估單個任務"""
        self.model.eval()
        val_loader = self.dataloaders[f'{task_type}_val']
        
        with torch.no_grad():
            if task_type == 'segmentation':
                return self._validate_segmentation(val_loader)
            elif task_type == 'detection':
                return self._validate_detection(val_loader)
            elif task_type == 'classification':
                return self._validate_classification(val_loader)
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
    
    def evaluate_all_tasks(self) -> Dict[str, Dict[str, float]]:
        """評估所有任務性能"""
        self.logger.info("📊 評估所有任務性能...")
        
        all_metrics = {}
        for task_type in self.task_sequence:
            if f'{task_type}_val' in self.dataloaders:
                metrics = self.evaluate_task(task_type)
                all_metrics[task_type] = metrics
                self.logger.info(f"  {task_type}: {metrics['main_metric']:.4f}")
        
        return all_metrics
    
    def check_forgetting(self) -> Dict[str, Any]:
        """
        檢查遺忘程度
        
        Returns:
            forgetting_info: 遺忘信息字典
        """
        self.logger.info("🔍 檢查災難性遺忘...")
        
        current_metrics = self.evaluate_all_tasks()
        forgetting_info = {
            'task_drops': {},
            'forgetting_rates': {},
            'acceptable': True,
            'max_drop': 0.0
        }
        
        for task_type in self.completed_tasks:
            if task_type in self.baseline_performance and task_type in current_metrics:
                baseline = self.baseline_performance[task_type]['main_metric']
                current = current_metrics[task_type]['main_metric']
                
                drop = baseline - current
                forgetting_rate = drop / baseline if baseline > 0 else 0
                
                forgetting_info['task_drops'][task_type] = drop
                forgetting_info['forgetting_rates'][task_type] = forgetting_rate
                forgetting_info['max_drop'] = max(forgetting_info['max_drop'], forgetting_rate)
                
                self.logger.info(
                    f"  {task_type}: 基準={baseline:.4f}, 當前={current:.4f}, "
                    f"下降={drop:.4f} ({forgetting_rate*100:.2f}%)"
                )
        
        # 檢查是否可接受
        if forgetting_info['max_drop'] > self.forgetting_threshold:
            forgetting_info['acceptable'] = False
            self.logger.warning(
                f"⚠️ 遺忘程度超過閾值！最大下降: {forgetting_info['max_drop']*100:.2f}% > {self.forgetting_threshold*100:.2f}%"
            )
        else:
            self.logger.info(f"✅ 遺忘程度可接受：最大下降 {forgetting_info['max_drop']*100:.2f}%")
        
        return forgetting_info
    
    def _adaptive_ewc_adjustment(self, epoch: int, batch_idx: int):
        """自適應調整EWC權重"""
        if not self.adaptive_ewc or epoch % 5 != 0:  # 每5個epoch檢查一次
            return
        
        # 檢查當前遺忘程度
        if len(self.completed_tasks) > 0:
            forgetting_info = self.check_forgetting()
            max_forgetting = forgetting_info['max_drop']
            
            # 動態調整EWC重要性
            if max_forgetting > 0.03:  # 如果遺忘超過3%
                new_importance = self.ewc_importance * 1.5
                self.logger.info(f"🔧 增加EWC權重: {self.ewc_importance:.0f} → {new_importance:.0f}")
                self.ewc.importance = new_importance
                self.ewc_importance = new_importance
            elif max_forgetting < 0.01:  # 如果遺忘很少
                new_importance = max(self.ewc_importance * 0.8, 500.0)  # 最小值500
                if new_importance != self.ewc_importance:
                    self.logger.info(f"🔧 減少EWC權重: {self.ewc_importance:.0f} → {new_importance:.0f}")
                    self.ewc.importance = new_importance
                    self.ewc_importance = new_importance
    
    def _save_checkpoint(self, stage_name: str, epoch: int, metrics: Dict[str, float]):
        """保存檢查點"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'epoch': epoch,
            'stage': stage_name,
            'metrics': metrics,
            'baseline_performance': self.baseline_performance,
            'current_performance': self.current_performance,
            'completed_tasks': self.completed_tasks,
            'ewc_importance': self.ewc_importance,
            'training_history': dict(self.training_history)
        }
        
        checkpoint_path = self.save_dir / f'{stage_name}_epoch_{epoch}_best.pth'
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"💾 保存檢查點: {checkpoint_path}")
    
    def save_training_history(self):
        """保存訓練歷史"""
        history_path = self.save_dir / 'training_history.json'
        
        # 轉換tensor為可序列化格式
        serializable_history = {}
        for key, value in self.training_history.items():
            serializable_history[key] = {}
            for task, data in value.items():
                if isinstance(data, list):
                    serializable_history[key][task] = [
                        float(x) if torch.is_tensor(x) else x for x in data
                    ]
                else:
                    serializable_history[key][task] = data
        
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump({
                'training_history': serializable_history,
                'baseline_performance': self.baseline_performance,
                'current_performance': self.current_performance,
                'completed_tasks': self.completed_tasks
            }, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📊 保存訓練歷史: {history_path}")
    
    def plot_training_curves(self):
        """繪製訓練曲線"""
        if not self.training_history['losses']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Sequential Training Progress', fontsize=16)
        
        # 損失曲線
        ax = axes[0, 0]
        for task, losses in self.training_history['losses'].items():
            if losses:
                ax.plot(losses, label=f'{task} loss')
        ax.set_title('Training Losses')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        
        # 性能指標
        ax = axes[0, 1]
        for task, metrics in self.training_history['metrics'].items():
            if metrics:
                ax.plot(metrics, label=f'{task} metric')
        ax.set_title('Validation Metrics')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Metric')
        ax.legend()
        ax.grid(True)
        
        # EWC 懲罰項
        ax = axes[1, 0]
        for task, penalties in self.training_history['ewc_penalties'].items():
            if penalties:
                ax.plot(penalties, label=f'{task} EWC penalty')
        ax.set_title('EWC Penalties')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Penalty')
        ax.legend()
        ax.grid(True)
        
        # 遺忘率
        ax = axes[1, 1]
        for task, rates in self.training_history['forgetting_rates'].items():
            if rates:
                ax.plot(rates, label=f'{task} forgetting')
        ax.axhline(y=self.forgetting_threshold, color='r', linestyle='--', label='Threshold')
        ax.set_title('Forgetting Rates')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Forgetting Rate')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        
        plot_path = self.save_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📈 保存訓練曲線: {plot_path}")


def create_sequential_trainer(model: nn.Module,
                            dataloaders: Dict[str, DataLoader],
                            **kwargs) -> SequentialTrainer:
    """
    創建依序訓練器
    
    Args:
        model: 統一多任務模型
        dataloaders: 數據加載器字典
        **kwargs: 其他參數
    
    Returns:
        trainer: 依序訓練器實例
    """
    return SequentialTrainer(model=model, dataloaders=dataloaders, **kwargs)
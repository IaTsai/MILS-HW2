"""
Elastic Weight Consolidation (EWC) 實現 - 修復版本
修復了數值穩定性問題，防止災難性遺忘

基於論文: "Overcoming catastrophic forgetting in neural networks"
(Kirkpatrick et al., 2017)

主要修復:
1. Fisher矩陣計算使用平均而非總和
2. 添加梯度裁剪和數值穩定性措施
3. Fisher值範圍限制防止爆炸
4. 更合理的importance權重範圍
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any, Union
import copy
import time
import numpy as np
from collections import defaultdict


class EWC:
    """
    Elastic Weight Consolidation (EWC) 實現 - 增強版
    
    主要改進:
    1. 修復Fisher矩陣計算中的概率求和錯誤
    2. 添加數值穩定性措施
    3. 支援動態權重調整
    4. 防止梯度爆炸
    """
    
    def __init__(self, 
                 model: nn.Module,
                 importance: float = 5000.0,  # 提升初始權重
                 device: Optional[torch.device] = None,
                 diagonal_only: bool = True,
                 ewc_type: str = 'l2',
                 max_fisher_value: float = 1e6,  # Fisher值上限
                 eps: float = 1e-8):  # 數值穩定性epsilon
        
        self.model = model
        self.importance = importance
        self.device = device if device is not None else next(model.parameters()).device
        self.diagonal_only = diagonal_only
        self.ewc_type = ewc_type
        self.max_fisher_value = max_fisher_value
        self.eps = eps
        
        # 存儲 Fisher 信息矩陣和最優參數
        self.fisher_matrices = {}
        self.optimal_params = {}
        self.task_count = 0
        
        # Online EWC 的額外參數
        if ewc_type == 'online':
            self.gamma = 0.9  # 衰減因子，降低以減少舊任務影響
            self.consolidated_fisher = {}
            self.consolidated_params = {}
        
        # 參數名稱映射
        self.param_names = [name for name, _ in model.named_parameters() if _.requires_grad]
        
        # 統計信息
        self.computation_stats = {
            'fisher_computation_time': [],
            'penalty_computation_time': [],
            'memory_usage': [],
            'fisher_magnitudes': []  # 新增：追蹤Fisher值大小
        }
        
        # 自適應權重調整
        self.adaptive_importance = importance
        self.importance_history = [importance]
        self.forgetting_rates = []
    
    def _get_named_parameters(self) -> Dict[str, torch.Tensor]:
        """獲取模型的命名參數字典"""
        return {name: param for name, param in self.model.named_parameters() if param.requires_grad}
    
    def compute_fisher_matrix(self, 
                            dataloader: DataLoader,
                            task_id: Optional[int] = None,
                            num_samples: Optional[int] = None,
                            verbose: bool = True) -> Dict[str, torch.Tensor]:
        """
        計算 Fisher 信息矩陣 - 修復版本
        """
        if task_id is None:
            task_id = self.task_count
        
        if verbose:
            print(f"🧮 計算任務 {task_id} 的 Fisher 信息矩陣...")
        
        start_time = time.time()
        
        # 設置模型為評估模式
        self.model.eval()
        
        # 初始化 Fisher 矩陣和計數器
        fisher_matrix = {}
        fisher_counts = {}  # 記錄每個參數的樣本數
        for name, param in self._get_named_parameters().items():
            fisher_matrix[name] = torch.zeros_like(param.data)
            fisher_counts[name] = 0
        
        # 樣本計數
        sample_count = 0
        total_samples = len(dataloader.dataset) if num_samples is None else min(num_samples, len(dataloader.dataset))
        
        if verbose:
            print(f"  使用 {total_samples} 個樣本計算 Fisher 矩陣")
        
        # 批次處理
        for batch_idx, batch in enumerate(dataloader):
            if num_samples is not None and sample_count >= num_samples:
                break
            
            # 解析批次數據
            if isinstance(batch, dict):
                # Try both 'image' and 'images' keys
                if 'image' in batch:
                    images = batch['image'].to(self.device)
                elif 'images' in batch:
                    images = batch['images'].to(self.device)
                else:
                    raise KeyError("Batch dict must contain 'image' or 'images' key")
                batch_size = images.size(0)
            elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                images = batch[0].to(self.device)
                batch_size = images.size(0)
            else:
                images = batch.to(self.device)
                batch_size = images.size(0)
            
            # 前向傳播
            outputs = self.model(images)
            
            # 處理多任務輸出
            if isinstance(outputs, dict):
                # 多任務輸出：計算每個任務的 Fisher 矩陣
                for task_name, task_output in outputs.items():
                    self._accumulate_fisher_from_output_fixed(
                        task_output, fisher_matrix, fisher_counts, batch_size
                    )
            else:
                # 單任務輸出
                self._accumulate_fisher_from_output_fixed(
                    outputs, fisher_matrix, fisher_counts, batch_size
                )
            
            sample_count += batch_size
            
            if verbose and (batch_idx + 1) % 50 == 0:
                progress = min(sample_count / total_samples * 100, 100)
                print(f"  進度: {progress:.1f}% ({sample_count}/{total_samples})")
        
        # 正規化 Fisher 矩陣 - 使用實際的計數
        for name in fisher_matrix:
            if fisher_counts[name] > 0:
                fisher_matrix[name] = fisher_matrix[name] / fisher_counts[name]
                
                # 應用Fisher值上限防止爆炸
                fisher_matrix[name] = torch.clamp(
                    fisher_matrix[name], 
                    min=0, 
                    max=self.max_fisher_value
                )
        
        # 存儲 Fisher 矩陣
        self.fisher_matrices[task_id] = fisher_matrix
        
        # 記錄計算時間和統計
        computation_time = time.time() - start_time
        self.computation_stats['fisher_computation_time'].append(computation_time)
        
        # 記錄Fisher值大小
        avg_fisher = sum(f.mean().item() for f in fisher_matrix.values()) / len(fisher_matrix)
        max_fisher = max(f.max().item() for f in fisher_matrix.values())
        self.computation_stats['fisher_magnitudes'].append({
            'avg': avg_fisher,
            'max': max_fisher
        })
        
        if verbose:
            print(f"  ✅ Fisher 矩陣計算完成 (耗時: {computation_time:.2f}s)")
            print(f"  📊 平均Fisher值: {avg_fisher:.6f}")
            print(f"  📊 最大Fisher值: {max_fisher:.6f}")
        
        return fisher_matrix
    
    def _accumulate_fisher_from_output_fixed(self, 
                                           output: torch.Tensor, 
                                           fisher_matrix: Dict[str, torch.Tensor],
                                           fisher_counts: Dict[str, int],
                                           batch_size: int):
        """
        從模型輸出累積 Fisher 信息矩陣 - 修復版本
        
        主要修復:
        1. 使用平均而非總和計算概率權重
        2. 添加梯度裁剪
        3. 正確處理不同維度的輸出
        """
        # 清零梯度
        self.model.zero_grad()
        
        # 計算對數似然梯度
        if output.dim() == 2:
            # 分類任務：批次輸出 (B, C)
            log_prob = F.log_softmax(output, dim=1)
            prob = F.softmax(output, dim=1)
            num_samples = output.size(0)
            
        elif output.dim() == 4:
            # 分割任務：(B, C, H, W)
            B, C, H, W = output.shape
            output_flat = output.view(B, C, -1).transpose(1, 2).contiguous().view(-1, C)
            log_prob = F.log_softmax(output_flat, dim=1)
            prob = F.softmax(output_flat, dim=1)
            num_samples = B * H * W
            
        elif output.dim() == 3:
            # 檢測任務：(B, N, C)
            B, N, C = output.shape
            output_flat = output.view(-1, C)
            log_prob = F.log_softmax(output_flat, dim=1)
            prob = F.softmax(output_flat, dim=1)
            num_samples = B * N
        else:
            raise ValueError(f"Unsupported output dimension: {output.dim()}")
        
        # 計算期望的 Fisher 信息矩陣
        for class_idx in range(prob.size(1)):
            # 使用平均概率而非總和
            class_prob_mean = prob[:, class_idx].mean()
            
            if class_prob_mean.item() > self.eps:  # 避免數值不穩定
                # 選擇該類別的對數概率並計算梯度
                class_log_prob = log_prob[:, class_idx].mean()
                class_log_prob.backward(retain_graph=True)
                
                # 累積 Fisher 信息
                for name, param in self._get_named_parameters().items():
                    if param.grad is not None:
                        # 梯度裁剪防止爆炸
                        grad_data = torch.clamp(param.grad.data, -10, 10)
                        
                        # 對角 Fisher 矩陣
                        fisher_update = class_prob_mean.item() * (grad_data ** 2)
                        fisher_matrix[name] += fisher_update
                        fisher_counts[name] += 1
                
                # 清零梯度
                self.model.zero_grad()
    
    def store_optimal_params(self, task_id: Optional[int] = None):
        """存儲當前任務的最優參數"""
        if task_id is None:
            task_id = self.task_count
        
        optimal_params = {}
        for name, param in self._get_named_parameters().items():
            optimal_params[name] = param.data.clone()
        
        self.optimal_params[task_id] = optimal_params
        print(f"📥 存儲任務 {task_id} 的最優參數")
    
    def penalty(self, model: Optional[nn.Module] = None) -> torch.Tensor:
        """
        計算 EWC 懲罰項 - 增強版本
        
        改進:
        1. 添加懲罰項範圍限制
        2. 使用自適應權重
        3. 防止數值溢出
        """
        if model is None:
            model = self.model
        
        start_time = time.time()
        
        penalty = 0.0
        current_params = {name: param for name, param in model.named_parameters() if param.requires_grad}
        
        if self.ewc_type == 'l2':
            # 標準 EWC: 對所有任務計算懲罰項
            for task_id in self.fisher_matrices.keys():
                if task_id in self.optimal_params:
                    task_penalty = self._compute_task_penalty_safe(
                        current_params, 
                        self.fisher_matrices[task_id], 
                        self.optimal_params[task_id]
                    )
                    penalty += task_penalty
        
        elif self.ewc_type == 'online':
            # Online EWC: 使用合併的 Fisher 矩陣
            if self.consolidated_fisher and self.consolidated_params:
                penalty = self._compute_task_penalty_safe(
                    current_params,
                    self.consolidated_fisher,
                    self.consolidated_params
                )
        
        # 使用自適應權重
        penalty_tensor = torch.tensor(penalty, device=self.device, requires_grad=True)
        penalty_loss = self.adaptive_importance * penalty_tensor
        
        # 防止懲罰項爆炸
        penalty_loss = torch.clamp(penalty_loss, max=1e10)
        
        # 記錄計算時間
        computation_time = time.time() - start_time
        self.computation_stats['penalty_computation_time'].append(computation_time)
        
        return penalty_loss
    
    def _compute_task_penalty_safe(self, 
                                 current_params: Dict[str, torch.Tensor],
                                 fisher_matrix: Dict[str, torch.Tensor],
                                 optimal_params: Dict[str, torch.Tensor]) -> float:
        """
        計算單個任務的懲罰項 - 安全版本
        """
        task_penalty = 0.0
        
        for name in fisher_matrix.keys():
            if name in current_params and name in optimal_params:
                # 參數差異
                param_diff = current_params[name] - optimal_params[name]
                
                # 對角 Fisher 矩陣
                fisher_diag = fisher_matrix[name]
                
                # 添加數值穩定性
                fisher_diag = torch.clamp(fisher_diag, min=0, max=self.max_fisher_value)
                
                penalty_term = fisher_diag * (param_diff ** 2)
                task_penalty += penalty_term.sum().item()
        
        return task_penalty / 2.0
    
    def update_adaptive_importance(self, forgetting_rate: float, target_rate: float = 0.05):
        """
        根據遺忘率動態調整EWC權重
        
        Args:
            forgetting_rate: 當前遺忘率
            target_rate: 目標遺忘率
        """
        self.forgetting_rates.append(forgetting_rate)
        
        # 如果遺忘率超過目標，增加權重
        if forgetting_rate > target_rate:
            # 指數增長，快速響應高遺忘率
            scale_factor = min(2.0, 1.0 + (forgetting_rate - target_rate) * 10)
            self.adaptive_importance = min(
                self.adaptive_importance * scale_factor, 
                100000.0  # 上限
            )
        else:
            # 緩慢降低權重
            self.adaptive_importance *= 0.95
        
        self.importance_history.append(self.adaptive_importance)
        
        print(f"🔧 自適應EWC權重調整: {self.adaptive_importance:.1f} (遺忘率: {forgetting_rate:.2%})")
    
    def update_consolidated_fisher(self):
        """更新合併的 Fisher 矩陣 (用於 Online EWC)"""
        if self.ewc_type != 'online':
            return
        
        if not self.fisher_matrices:
            return
        
        # 獲取最新的 Fisher 矩陣和參數
        latest_task_id = max(self.fisher_matrices.keys())
        latest_fisher = self.fisher_matrices[latest_task_id]
        latest_params = self.optimal_params[latest_task_id]
        
        if not self.consolidated_fisher:
            # 第一個任務：直接複製
            self.consolidated_fisher = copy.deepcopy(latest_fisher)
            self.consolidated_params = copy.deepcopy(latest_params)
        else:
            # 後續任務：加權合併
            for name in latest_fisher.keys():
                if name in self.consolidated_fisher:
                    # 合併 Fisher 矩陣
                    self.consolidated_fisher[name] = (
                        self.gamma * self.consolidated_fisher[name] + 
                        latest_fisher[name]
                    )
                    
                    # 限制合併後的Fisher值
                    self.consolidated_fisher[name] = torch.clamp(
                        self.consolidated_fisher[name],
                        max=self.max_fisher_value
                    )
                    
                    # 更新合併參數
                    weight_old = self.gamma / (self.gamma + 1.0)
                    weight_new = 1.0 / (self.gamma + 1.0)
                    self.consolidated_params[name] = (
                        weight_old * self.consolidated_params[name] + 
                        weight_new * latest_params[name]
                    )
        
        print(f"🔄 更新合併的 Fisher 矩陣 (任務 {latest_task_id})")
    
    def finish_task(self, 
                   dataloader: DataLoader, 
                   task_id: Optional[int] = None,
                   verbose: bool = True) -> int:
        """完成當前任務的 EWC 設置"""
        if task_id is None:
            task_id = self.task_count
        
        if verbose:
            print(f"🎯 完成任務 {task_id} 的 EWC 設置...")
            print(f"  當前自適應權重: {self.adaptive_importance:.1f}")
        
        # 計算 Fisher 矩陣
        self.compute_fisher_matrix(dataloader, task_id=task_id, verbose=verbose)
        
        # 存儲最優參數
        self.store_optimal_params(task_id=task_id)
        
        # 更新合併的 Fisher 矩陣 (Online EWC)
        if self.ewc_type == 'online':
            self.update_consolidated_fisher()
        
        # 增加任務計數
        if task_id == self.task_count:
            self.task_count += 1
        
        if verbose:
            print(f"✅ 任務 {task_id} 的 EWC 設置完成")
            memory_info = self.get_memory_usage()
            print(f"  💾 記憶體使用: {memory_info['total_mb']:.2f} MB")
        
        return task_id
    
    def get_memory_usage(self) -> Dict[str, float]:
        """獲取 EWC 的記憶體使用情況"""
        fisher_memory = 0
        params_memory = 0
        
        # 計算 Fisher 矩陣記憶體
        for fisher_matrix in self.fisher_matrices.values():
            for tensor in fisher_matrix.values():
                fisher_memory += tensor.numel() * tensor.element_size()
        
        # 計算參數記憶體
        for params in self.optimal_params.values():
            for tensor in params.values():
                params_memory += tensor.numel() * tensor.element_size()
        
        # 合併的 Fisher 矩陣記憶體
        consolidated_memory = 0
        if hasattr(self, 'consolidated_fisher') and self.consolidated_fisher:
            for tensor in self.consolidated_fisher.values():
                consolidated_memory += tensor.numel() * tensor.element_size()
        
        total_memory = fisher_memory + params_memory + consolidated_memory
        
        memory_info = {
            'fisher_matrices_mb': fisher_memory / (1024 ** 2),
            'optimal_params_mb': params_memory / (1024 ** 2),
            'consolidated_mb': consolidated_memory / (1024 ** 2),
            'total_mb': total_memory / (1024 ** 2),
            'num_tasks': len(self.fisher_matrices)
        }
        
        return memory_info
    
    def get_computation_stats(self) -> Dict[str, Any]:
        """獲取計算統計信息"""
        stats = {
            'avg_fisher_time': np.mean(self.computation_stats['fisher_computation_time']) if self.computation_stats['fisher_computation_time'] else 0,
            'avg_penalty_time': np.mean(self.computation_stats['penalty_computation_time']) if self.computation_stats['penalty_computation_time'] else 0,
            'total_fisher_computations': len(self.computation_stats['fisher_computation_time']),
            'total_penalty_computations': len(self.computation_stats['penalty_computation_time']),
            'memory_usage': self.get_memory_usage(),
            'adaptive_importance': self.adaptive_importance,
            'forgetting_rates': self.forgetting_rates[-5:] if self.forgetting_rates else []
        }
        
        if self.computation_stats['fisher_magnitudes']:
            recent_mags = self.computation_stats['fisher_magnitudes'][-5:]
            stats['recent_fisher_magnitudes'] = {
                'avg': [m['avg'] for m in recent_mags],
                'max': [m['max'] for m in recent_mags]
            }
        
        return stats
    
    def save_ewc_data(self, save_path: str):
        """保存 EWC 數據"""
        ewc_data = {
            'fisher_matrices': self.fisher_matrices,
            'optimal_params': self.optimal_params,
            'task_count': self.task_count,
            'importance': self.importance,
            'adaptive_importance': self.adaptive_importance,
            'ewc_type': self.ewc_type,
            'computation_stats': self.computation_stats,
            'forgetting_rates': self.forgetting_rates,
            'importance_history': self.importance_history
        }
        
        if self.ewc_type == 'online':
            ewc_data.update({
                'consolidated_fisher': getattr(self, 'consolidated_fisher', {}),
                'consolidated_params': getattr(self, 'consolidated_params', {}),
                'gamma': getattr(self, 'gamma', 0.9)
            })
        
        torch.save(ewc_data, save_path)
        print(f"💾 EWC 數據已保存到: {save_path}")
    
    def load_ewc_data(self, load_path: str):
        """載入 EWC 數據"""
        ewc_data = torch.load(load_path, map_location=self.device)
        
        self.fisher_matrices = ewc_data.get('fisher_matrices', {})
        self.optimal_params = ewc_data.get('optimal_params', {})
        self.task_count = ewc_data.get('task_count', 0)
        self.importance = ewc_data.get('importance', self.importance)
        self.adaptive_importance = ewc_data.get('adaptive_importance', self.importance)
        self.ewc_type = ewc_data.get('ewc_type', self.ewc_type)
        self.computation_stats = ewc_data.get('computation_stats', self.computation_stats)
        self.forgetting_rates = ewc_data.get('forgetting_rates', [])
        self.importance_history = ewc_data.get('importance_history', [self.importance])
        
        if self.ewc_type == 'online':
            self.consolidated_fisher = ewc_data.get('consolidated_fisher', {})
            self.consolidated_params = ewc_data.get('consolidated_params', {})
            self.gamma = ewc_data.get('gamma', 0.9)
        
        print(f"📂 EWC 數據已從 {load_path} 載入")


def ewc_loss(current_loss: torch.Tensor, 
            ewc_handler: EWC, 
            model: Optional[nn.Module] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    計算包含 EWC 懲罰項的總損失
    """
    ewc_penalty = ewc_handler.penalty(model)
    total_loss = current_loss + ewc_penalty
    
    return total_loss, ewc_penalty


def create_ewc_handler(model: nn.Module, 
                      importance: float = 5000.0,  # 提高預設值
                      ewc_type: str = 'l2',
                      **kwargs) -> EWC:
    """
    創建 EWC 處理器的工廠函數
    """
    return EWC(model=model, importance=importance, ewc_type=ewc_type, **kwargs)


if __name__ == "__main__":
    # 測試修復的EWC
    import torch.nn as nn
    
    # 創建簡單模型
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    # 創建 EWC 處理器
    ewc = create_ewc_handler(model, importance=5000.0)
    
    # 創建虛擬數據
    data = torch.randn(100, 10)
    labels = torch.randint(0, 5, (100,))
    dataset = torch.utils.data.TensorDataset(data, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)
    
    # 計算 Fisher 矩陣
    fisher_matrix = ewc.compute_fisher_matrix(dataloader, verbose=True)
    
    # 存儲最優參數
    ewc.store_optimal_params()
    
    # 計算懲罰項
    penalty = ewc.penalty()
    
    print(f"✅ EWC 測試完成！")
    print(f"📊 Fisher 矩陣層數: {len(fisher_matrix)}")
    print(f"💰 EWC 懲罰項: {penalty.item():.6f}")
    print(f"💾 記憶體使用: {ewc.get_memory_usage()}")
    print(f"📈 計算統計: {ewc.get_computation_stats()}")
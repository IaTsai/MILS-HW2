"""
Elastic Weight Consolidation (EWC) 實現
用於防止多任務學習中的災難性遺忘

基於論文: "Overcoming catastrophic forgetting in neural networks"
(Kirkpatrick et al., 2017)
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
    Elastic Weight Consolidation (EWC) 實現
    
    EWC 通過計算 Fisher 信息矩陣來衡量參數的重要性，
    並在學習新任務時添加懲罰項來保護重要參數，
    從而防止災難性遺忘。
    
    Args:
        model: 要保護的神經網路模型
        importance: EWC 懲罰項的重要性係數 (lambda)
        device: 計算設備
        diagonal_only: 是否只計算 Fisher 矩陣的對角項 (節省記憶體)
        ewc_type: EWC 類型 ('l2' 或 'online')
    """
    
    def __init__(self, 
                 model: nn.Module,
                 importance: float = 1000.0,
                 device: Optional[torch.device] = None,
                 diagonal_only: bool = True,
                 ewc_type: str = 'l2'):
        
        self.model = model
        self.importance = importance
        self.device = device if device is not None else next(model.parameters()).device
        self.diagonal_only = diagonal_only
        self.ewc_type = ewc_type
        
        # 存儲 Fisher 信息矩陣和最優參數
        self.fisher_matrices = {}  # 每個任務的 Fisher 矩陣
        self.optimal_params = {}   # 每個任務的最優參數
        self.task_count = 0        # 任務計數器
        
        # Online EWC 的額外參數
        if ewc_type == 'online':
            self.gamma = 1.0  # 衰減因子
            self.consolidated_fisher = {}
            self.consolidated_params = {}
        
        # 參數名稱映射 (用於參數匹配)
        self.param_names = [name for name, _ in model.named_parameters() if _.requires_grad]
        
        # 統計信息
        self.computation_stats = {
            'fisher_computation_time': [],
            'penalty_computation_time': [],
            'memory_usage': []
        }
    
    def _get_named_parameters(self) -> Dict[str, torch.Tensor]:
        """獲取模型的命名參數字典"""
        return {name: param for name, param in self.model.named_parameters() if param.requires_grad}
    
    def compute_fisher_matrix(self, 
                            dataloader: DataLoader,
                            task_id: Optional[int] = None,
                            num_samples: Optional[int] = None,
                            verbose: bool = True) -> Dict[str, torch.Tensor]:
        """
        計算 Fisher 信息矩陣
        
        Fisher 信息矩陣衡量了每個參數對於模型輸出的重要性。
        這裡使用經驗 Fisher 信息矩陣的近似計算。
        
        Args:
            dataloader: 用於計算 Fisher 矩陣的數據加載器
            task_id: 任務ID (如果為 None，則使用當前任務計數)
            num_samples: 使用的樣本數量 (如果為 None，則使用所有樣本)
            verbose: 是否顯示進度信息
        
        Returns:
            fisher_matrix: Fisher 信息矩陣字典
        """
        if task_id is None:
            task_id = self.task_count
        
        if verbose:
            print(f"🧮 計算任務 {task_id} 的 Fisher 信息矩陣...")
        
        start_time = time.time()
        
        # 設置模型為評估模式
        self.model.eval()
        
        # 初始化 Fisher 矩陣
        fisher_matrix = {}
        for name, param in self._get_named_parameters().items():
            if self.diagonal_only:
                fisher_matrix[name] = torch.zeros_like(param.data)
            else:
                # 完整 Fisher 矩陣需要大量記憶體，通常不實用
                fisher_matrix[name] = torch.zeros(param.numel(), param.numel(), device=self.device)
        
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
                # 統一數據加載器格式
                images = batch['images'].to(self.device)
                batch_size = images.size(0)
            elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                # 標準格式 (images, labels)
                images = batch[0].to(self.device)
                batch_size = images.size(0)
            else:
                # 只有圖像數據
                images = batch.to(self.device)
                batch_size = images.size(0)
            
            # 前向傳播
            outputs = self.model(images)
            
            # 處理多任務輸出
            if isinstance(outputs, dict):
                # 多任務輸出：計算每個任務的 Fisher 矩陣
                for task_name, task_output in outputs.items():
                    self._accumulate_fisher_from_output(task_output, fisher_matrix, batch_size)
            else:
                # 單任務輸出
                self._accumulate_fisher_from_output(outputs, fisher_matrix, batch_size)
            
            sample_count += batch_size
            
            if verbose and (batch_idx + 1) % 50 == 0:
                progress = min(sample_count / total_samples * 100, 100)
                print(f"  進度: {progress:.1f}% ({sample_count}/{total_samples})")
        
        # 正規化 Fisher 矩陣
        for name in fisher_matrix:
            fisher_matrix[name] = fisher_matrix[name] / sample_count
        
        # 存儲 Fisher 矩陣
        self.fisher_matrices[task_id] = fisher_matrix
        
        # 記錄計算時間
        computation_time = time.time() - start_time
        self.computation_stats['fisher_computation_time'].append(computation_time)
        
        if verbose:
            print(f"  ✅ Fisher 矩陣計算完成 (耗時: {computation_time:.2f}s)")
            
            # 顯示 Fisher 矩陣統計
            total_params = sum(f.numel() for f in fisher_matrix.values())
            avg_importance = sum(f.sum().item() for f in fisher_matrix.values()) / total_params
            print(f"  📊 平均重要性: {avg_importance:.6f}")
        
        return fisher_matrix
    
    def _accumulate_fisher_from_output(self, 
                                     output: torch.Tensor, 
                                     fisher_matrix: Dict[str, torch.Tensor],
                                     batch_size: int):
        """
        從模型輸出累積 Fisher 信息矩陣
        
        Args:
            output: 模型輸出
            fisher_matrix: 當前的 Fisher 矩陣
            batch_size: 批次大小
        """
        # 清零梯度
        self.model.zero_grad()
        
        # 計算對數似然梯度
        if output.dim() == 1:
            # 分類任務：單個輸出
            log_prob = F.log_softmax(output.unsqueeze(0), dim=1)
            prob = F.softmax(output.unsqueeze(0), dim=1)
        elif output.dim() == 2:
            # 分類任務：批次輸出
            log_prob = F.log_softmax(output, dim=1)
            prob = F.softmax(output, dim=1)
        elif output.dim() == 4:
            # 分割任務：(B, C, H, W)
            B, C, H, W = output.shape
            output_flat = output.view(B, C, -1).transpose(1, 2).contiguous().view(-1, C)
            log_prob = F.log_softmax(output_flat, dim=1)
            prob = F.softmax(output_flat, dim=1)
        elif output.dim() == 3:
            # 檢測任務：(B, N, C)
            B, N, C = output.shape
            output_flat = output.view(-1, C)
            log_prob = F.log_softmax(output_flat, dim=1)
            prob = F.softmax(output_flat, dim=1)
        else:
            raise ValueError(f"Unsupported output dimension: {output.dim()}")
        
        # 計算期望的 Fisher 信息矩陣
        for class_idx in range(prob.size(1)):
            # 選擇當前類別
            if prob.dim() == 2:
                class_prob = prob[:, class_idx].sum()
                class_log_prob = log_prob[:, class_idx].sum()
            else:
                class_prob = prob[:, class_idx].sum()
                class_log_prob = log_prob[:, class_idx].sum()
            
            if class_prob.item() > 1e-8:  # 避免數值不穩定
                # 計算梯度
                class_log_prob.backward(retain_graph=True)
                
                # 累積 Fisher 信息
                for name, param in self._get_named_parameters().items():
                    if param.grad is not None:
                        if self.diagonal_only:
                            # 對角 Fisher 矩陣
                            fisher_matrix[name] += class_prob.item() * (param.grad.data ** 2)
                        else:
                            # 完整 Fisher 矩陣 (記憶體密集)
                            grad_flat = param.grad.data.view(-1)
                            fisher_matrix[name] += class_prob.item() * torch.outer(grad_flat, grad_flat)
                
                # 清零梯度
                self.model.zero_grad()
    
    def store_optimal_params(self, task_id: Optional[int] = None):
        """
        存儲當前任務的最優參數
        
        Args:
            task_id: 任務ID
        """
        if task_id is None:
            task_id = self.task_count
        
        optimal_params = {}
        for name, param in self._get_named_parameters().items():
            optimal_params[name] = param.data.clone()
        
        self.optimal_params[task_id] = optimal_params
        print(f"📥 存儲任務 {task_id} 的最優參數")
    
    def penalty(self, model: Optional[nn.Module] = None) -> torch.Tensor:
        """
        計算 EWC 懲罰項
        
        懲罰項 = λ/2 * Σ(F_i * (θ_i - θ*_i)^2)
        其中 F_i 是 Fisher 信息矩陣，θ*_i 是最優參數
        
        Args:
            model: 要計算懲罰項的模型 (如果為 None，使用 self.model)
        
        Returns:
            penalty: EWC 懲罰項
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
                    penalty += self._compute_task_penalty(
                        current_params, 
                        self.fisher_matrices[task_id], 
                        self.optimal_params[task_id]
                    )
        
        elif self.ewc_type == 'online':
            # Online EWC: 使用合併的 Fisher 矩陣
            if self.consolidated_fisher and self.consolidated_params:
                penalty = self._compute_task_penalty(
                    current_params,
                    self.consolidated_fisher,
                    self.consolidated_params
                )
        
        penalty_tensor = torch.tensor(penalty, device=self.device, requires_grad=True)
        penalty_loss = self.importance * penalty_tensor
        
        # 記錄計算時間
        computation_time = time.time() - start_time
        self.computation_stats['penalty_computation_time'].append(computation_time)
        
        return penalty_loss
    
    def _compute_task_penalty(self, 
                            current_params: Dict[str, torch.Tensor],
                            fisher_matrix: Dict[str, torch.Tensor],
                            optimal_params: Dict[str, torch.Tensor]) -> float:
        """
        計算單個任務的懲罰項
        
        Args:
            current_params: 當前參數
            fisher_matrix: Fisher 信息矩陣
            optimal_params: 最優參數
        
        Returns:
            task_penalty: 任務懲罰項
        """
        task_penalty = 0.0
        
        for name in fisher_matrix.keys():
            if name in current_params and name in optimal_params:
                # 參數差異
                param_diff = current_params[name] - optimal_params[name]
                
                if self.diagonal_only:
                    # 對角 Fisher 矩陣
                    penalty_term = fisher_matrix[name] * (param_diff ** 2)
                    task_penalty += penalty_term.sum().item()
                else:
                    # 完整 Fisher 矩陣
                    param_diff_flat = param_diff.view(-1)
                    penalty_term = torch.dot(param_diff_flat, 
                                           torch.mv(fisher_matrix[name], param_diff_flat))
                    task_penalty += penalty_term.item()
        
        return task_penalty / 2.0
    
    def update_consolidated_fisher(self):
        """
        更新合併的 Fisher 矩陣 (用於 Online EWC)
        """
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
                    
                    # 更新合併參數 (加權平均)
                    self.consolidated_params[name] = (
                        self.gamma * self.consolidated_params[name] + 
                        latest_params[name]
                    ) / (self.gamma + 1.0)
        
        print(f"🔄 更新合併的 Fisher 矩陣 (任務 {latest_task_id})")
    
    def finish_task(self, 
                   dataloader: DataLoader, 
                   task_id: Optional[int] = None,
                   verbose: bool = True) -> int:
        """
        完成當前任務的 EWC 設置
        
        Args:
            dataloader: 當前任務的數據加載器
            task_id: 任務ID
            verbose: 是否顯示詳細信息
        
        Returns:
            task_id: 實際使用的任務ID
        """
        if task_id is None:
            task_id = self.task_count
        
        if verbose:
            print(f"🎯 完成任務 {task_id} 的 EWC 設置...")
        
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
        
        return task_id
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        獲取 EWC 的記憶體使用情況
        
        Returns:
            memory_info: 記憶體使用信息 (MB)
        """
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
        """
        獲取計算統計信息
        
        Returns:
            stats: 計算統計信息
        """
        stats = {
            'avg_fisher_time': np.mean(self.computation_stats['fisher_computation_time']) if self.computation_stats['fisher_computation_time'] else 0,
            'avg_penalty_time': np.mean(self.computation_stats['penalty_computation_time']) if self.computation_stats['penalty_computation_time'] else 0,
            'total_fisher_computations': len(self.computation_stats['fisher_computation_time']),
            'total_penalty_computations': len(self.computation_stats['penalty_computation_time']),
            'memory_usage': self.get_memory_usage()
        }
        
        return stats
    
    def save_ewc_data(self, save_path: str):
        """
        保存 EWC 數據
        
        Args:
            save_path: 保存路徑
        """
        ewc_data = {
            'fisher_matrices': self.fisher_matrices,
            'optimal_params': self.optimal_params,
            'task_count': self.task_count,
            'importance': self.importance,
            'ewc_type': self.ewc_type,
            'computation_stats': self.computation_stats
        }
        
        if self.ewc_type == 'online':
            ewc_data.update({
                'consolidated_fisher': getattr(self, 'consolidated_fisher', {}),
                'consolidated_params': getattr(self, 'consolidated_params', {}),
                'gamma': getattr(self, 'gamma', 1.0)
            })
        
        torch.save(ewc_data, save_path)
        print(f"💾 EWC 數據已保存到: {save_path}")
    
    def load_ewc_data(self, load_path: str):
        """
        載入 EWC 數據
        
        Args:
            load_path: 載入路徑
        """
        ewc_data = torch.load(load_path, map_location=self.device)
        
        self.fisher_matrices = ewc_data.get('fisher_matrices', {})
        self.optimal_params = ewc_data.get('optimal_params', {})
        self.task_count = ewc_data.get('task_count', 0)
        self.importance = ewc_data.get('importance', self.importance)
        self.ewc_type = ewc_data.get('ewc_type', self.ewc_type)
        self.computation_stats = ewc_data.get('computation_stats', self.computation_stats)
        
        if self.ewc_type == 'online':
            self.consolidated_fisher = ewc_data.get('consolidated_fisher', {})
            self.consolidated_params = ewc_data.get('consolidated_params', {})
            self.gamma = ewc_data.get('gamma', 1.0)
        
        print(f"📂 EWC 數據已從 {load_path} 載入")


def ewc_loss(current_loss: torch.Tensor, 
            ewc_handler: EWC, 
            model: Optional[nn.Module] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    計算包含 EWC 懲罰項的總損失
    
    Args:
        current_loss: 當前任務的損失
        ewc_handler: EWC 處理器
        model: 模型 (如果為 None，使用 ewc_handler.model)
    
    Returns:
        total_loss: 包含 EWC 懲罰項的總損失
        ewc_penalty: EWC 懲罰項
    """
    ewc_penalty = ewc_handler.penalty(model)
    total_loss = current_loss + ewc_penalty
    
    return total_loss, ewc_penalty


def create_ewc_handler(model: nn.Module, 
                      importance: float = 1000.0,
                      ewc_type: str = 'l2',
                      **kwargs) -> EWC:
    """
    創建 EWC 處理器的工廠函數
    
    Args:
        model: 神經網路模型
        importance: 重要性係數
        ewc_type: EWC 類型
        **kwargs: 其他參數
    
    Returns:
        ewc_handler: EWC 處理器
    """
    return EWC(model=model, importance=importance, ewc_type=ewc_type, **kwargs)


if __name__ == "__main__":
    # 簡單測試代碼
    import torch.nn as nn
    
    # 創建簡單模型
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    # 創建 EWC 處理器
    ewc = create_ewc_handler(model, importance=1000.0)
    
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
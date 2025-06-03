import numpy as np
import torch
import json
from typing import Dict, List, Tuple, Optional
from itertools import product
import os
from datetime import datetime

class HyperparameterOptimizer:
    """超參數優化器，專注於降低災難性遺忘率"""
    
    def __init__(self):
        # 基於災難性遺忘評估調整參數空間
        self.param_space = {
            'learning_rate': {
                'classification': [1e-3, 2e-3, 5e-3, 1e-2],  # 重點提升分類學習率
                'segmentation': [5e-4, 1e-3, 2e-3],          # 分割微調
                'detection': [1e-4, 5e-4, 1e-3]              # 檢測保持穩定
            },
            'task_weights': {
                'classification': [3.0, 5.0, 8.0, 10.0],     # 大幅提高分類權重
                'segmentation': [1.5, 2.0, 2.5],             # 分割適度調整
                'detection': [0.3, 0.5, 0.8]                 # 檢測降低避免過度主導
            },
            'architecture_changes': {
                'remove_classification_batchnorm': True,      # 移除分類BatchNorm
                'classification_head_layers': [2, 3, 4],      # 調整分類頭部深度
                'dropout_rate': [0.1, 0.3, 0.5]              # 正則化策略
            },
            'training_params': {
                'epochs': [30, 50, 100],                      # 訓練輪數
                'batch_size': [8, 16, 32],                    # 批次大小
                'gradient_clip': [0.5, 1.0, 2.0],            # 梯度裁剪
                'label_smoothing': [0.0, 0.1, 0.2]          # 標籤平滑
            }
        }
        
        self.results_history = []
        self.best_config = None
        self.best_forgetting_rates = {
            'classification': float('inf'),
            'segmentation': float('inf'),
            'detection': float('inf')
        }
        
    def grid_search(self, model_builder, train_func, eval_func, max_trials=10):
        """
        執行網格搜索，尋找最佳超參數組合
        
        Args:
            model_builder: 構建模型的函數
            train_func: 訓練函數
            eval_func: 評估函數（返回遺忘率）
            max_trials: 最大試驗次數
        """
        # 生成參數組合
        param_combinations = self._generate_combinations()
        
        # 根據優先級排序（優先測試高分類學習率和權重的組合）
        param_combinations = self._prioritize_combinations(param_combinations)
        
        # 限制試驗次數
        param_combinations = param_combinations[:max_trials]
        
        print(f"🔍 開始超參數搜索，共 {len(param_combinations)} 個組合")
        
        for idx, params in enumerate(param_combinations):
            print(f"\n📊 試驗 {idx+1}/{len(param_combinations)}")
            print(f"參數配置: {json.dumps(params, indent=2)}")
            
            # 構建模型
            model = model_builder(params['architecture_changes'])
            
            # 訓練模型
            trained_model = train_func(
                model,
                learning_rates=params['learning_rate'],
                task_weights=params['task_weights'],
                **params['training_params']
            )
            
            # 評估遺忘率
            forgetting_rates = eval_func(trained_model)
            
            # 記錄結果
            result = {
                'params': params,
                'forgetting_rates': forgetting_rates,
                'timestamp': datetime.now().isoformat()
            }
            self.results_history.append(result)
            
            # 更新最佳配置
            if self._is_better(forgetting_rates):
                self.best_config = params
                self.best_forgetting_rates = forgetting_rates.copy()
                print(f"✅ 新的最佳配置！遺忘率: {forgetting_rates}")
                
            # 早停條件：所有任務遺忘率都 ≤5%
            if all(rate <= 5.0 for rate in forgetting_rates.values()):
                print(f"🎯 達到目標！所有任務遺忘率 ≤5%")
                break
                
        return self.best_config, self.best_forgetting_rates
    
    def suggest_improvements(self, current_results: Dict[str, float]) -> Dict[str, any]:
        """
        基於當前結果建議改進方向
        
        Args:
            current_results: 當前的遺忘率結果
            
        Returns:
            改進建議字典
        """
        suggestions = {
            'priority_tasks': [],
            'parameter_adjustments': {},
            'architecture_changes': {},
            'training_strategy': []
        }
        
        # 分析各任務的遺忘率
        for task, rate in current_results.items():
            if rate > 5.0:
                suggestions['priority_tasks'].append(task)
                
                if task == 'classification' and rate > 20:
                    # 分類任務嚴重遺忘
                    suggestions['parameter_adjustments'][task] = {
                        'learning_rate': 'increase_by_5x',
                        'task_weight': 'increase_to_8-10x',
                        'reason': '分類任務梯度被嚴重壓制'
                    }
                    suggestions['architecture_changes'][task] = {
                        'remove_batchnorm': True,
                        'increase_capacity': True,
                        'add_skip_connections': True
                    }
                    
                elif task == 'segmentation' and rate > 5:
                    # 分割任務輕微遺忘
                    suggestions['parameter_adjustments'][task] = {
                        'learning_rate': 'increase_by_1.5x',
                        'task_weight': 'increase_to_2x',
                        'reason': '分割任務需要微調'
                    }
                    
        # 訓練策略建議
        if len(suggestions['priority_tasks']) > 0:
            suggestions['training_strategy'].extend([
                '考慮使用任務特定的學習率調度',
                '增加高優先級任務的批次採樣頻率',
                '使用梯度累積來穩定訓練',
                '考慮預熱期單獨訓練問題任務'
            ])
            
        return suggestions
    
    def _generate_combinations(self) -> List[Dict]:
        """生成所有參數組合"""
        combinations = []
        
        # 提取各類參數
        lr_combos = list(product(
            self.param_space['learning_rate']['classification'],
            self.param_space['learning_rate']['segmentation'],
            self.param_space['learning_rate']['detection']
        ))
        
        weight_combos = list(product(
            self.param_space['task_weights']['classification'],
            self.param_space['task_weights']['segmentation'],
            self.param_space['task_weights']['detection']
        ))
        
        arch_combos = list(product(
            self.param_space['architecture_changes']['classification_head_layers'],
            self.param_space['architecture_changes']['dropout_rate']
        ))
        
        train_combos = list(product(
            self.param_space['training_params']['epochs'],
            self.param_space['training_params']['batch_size'],
            self.param_space['training_params']['gradient_clip'],
            self.param_space['training_params']['label_smoothing']
        ))
        
        # 組合所有參數
        for lr, weight, arch, train in product(lr_combos, weight_combos, arch_combos, train_combos):
            combo = {
                'learning_rate': {
                    'classification': lr[0],
                    'segmentation': lr[1],
                    'detection': lr[2]
                },
                'task_weights': {
                    'classification': weight[0],
                    'segmentation': weight[1],
                    'detection': weight[2]
                },
                'architecture_changes': {
                    'remove_classification_batchnorm': self.param_space['architecture_changes']['remove_classification_batchnorm'],
                    'classification_head_layers': arch[0],
                    'dropout_rate': arch[1]
                },
                'training_params': {
                    'epochs': train[0],
                    'batch_size': train[1],
                    'gradient_clip': train[2],
                    'label_smoothing': train[3]
                }
            }
            combinations.append(combo)
            
        return combinations
    
    def _prioritize_combinations(self, combinations: List[Dict]) -> List[Dict]:
        """根據優先級排序參數組合"""
        def priority_score(combo):
            # 優先考慮高分類學習率和權重的組合
            score = 0
            score += combo['learning_rate']['classification'] * 1000
            score += combo['task_weights']['classification'] * 100
            score += (50 - combo['training_params']['epochs']) * 0.1  # 優先短訓練
            return score
            
        return sorted(combinations, key=priority_score, reverse=True)
    
    def _is_better(self, new_rates: Dict[str, float]) -> bool:
        """判斷新的遺忘率是否更好"""
        # 優先級：首先確保所有任務 ≤5%，然後最小化總遺忘率
        current_violations = sum(1 for rate in self.best_forgetting_rates.values() if rate > 5.0)
        new_violations = sum(1 for rate in new_rates.values() if rate > 5.0)
        
        if new_violations < current_violations:
            return True
        elif new_violations == current_violations:
            return sum(new_rates.values()) < sum(self.best_forgetting_rates.values())
        return False
    
    def save_results(self, filepath: str):
        """保存優化結果"""
        results = {
            'best_config': self.best_config,
            'best_forgetting_rates': self.best_forgetting_rates,
            'all_results': self.results_history,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"💾 優化結果已保存至: {filepath}")


class AdaptiveLossWeighting:
    """自適應損失權重調整，專注於降低遺忘率"""
    
    def __init__(self):
        self.weights_history = []
        self.performance_history = []
        # 基於災難性遺忘目標設定權重
        self.target_forgetting_rates = {
            'classification': 5.0,  # 最大允許遺忘率
            'segmentation': 5.0,
            'detection': 5.0
        }
        
        # 初始權重（基於診斷結果）
        self.current_weights = {
            'classification': 5.0,  # 高權重幫助分類學習
            'segmentation': 2.0,    # 適中權重
            'detection': 0.5        # 低權重避免過度主導
        }
        
        # 權重調整參數
        self.adjustment_rate = 0.1
        self.min_weights = {'classification': 1.0, 'segmentation': 0.5, 'detection': 0.1}
        self.max_weights = {'classification': 10.0, 'segmentation': 5.0, 'detection': 2.0}
        
    def update_weights(self, current_losses: Dict[str, float], 
                      current_forgetting_rates: Dict[str, float]) -> Dict[str, float]:
        """
        基於當前損失和遺忘率更新權重
        
        Args:
            current_losses: 當前各任務的損失值
            current_forgetting_rates: 當前各任務的遺忘率
            
        Returns:
            更新後的權重
        """
        # 記錄歷史
        self.weights_history.append(self.current_weights.copy())
        self.performance_history.append({
            'losses': current_losses.copy(),
            'forgetting_rates': current_forgetting_rates.copy()
        })
        
        # 計算權重調整
        for task in self.current_weights:
            if task in current_forgetting_rates:
                forgetting_rate = current_forgetting_rates[task]
                target_rate = self.target_forgetting_rates[task]
                
                # 如果遺忘率超過目標，增加權重
                if forgetting_rate > target_rate:
                    # 調整幅度與超出程度成正比
                    adjustment = self.adjustment_rate * (forgetting_rate - target_rate) / target_rate
                    self.current_weights[task] *= (1 + adjustment)
                    
                # 如果遺忘率遠低於目標，可以適度降低權重
                elif forgetting_rate < target_rate * 0.5:
                    adjustment = self.adjustment_rate * 0.5
                    self.current_weights[task] *= (1 - adjustment)
                    
                # 限制權重範圍
                self.current_weights[task] = np.clip(
                    self.current_weights[task],
                    self.min_weights[task],
                    self.max_weights[task]
                )
                
        # 歸一化權重（可選）
        # total = sum(self.current_weights.values())
        # self.current_weights = {k: v/total * 3.0 for k, v in self.current_weights.items()}
        
        return self.current_weights
    
    def suggest_optimal_weights(self) -> Dict[str, float]:
        """基於歷史數據建議最佳權重組合"""
        if len(self.performance_history) < 5:
            # 數據不足，返回當前權重
            return self.current_weights
            
        # 找出遺忘率最低的配置
        best_idx = -1
        best_total_forgetting = float('inf')
        
        for i, perf in enumerate(self.performance_history):
            total_forgetting = sum(perf['forgetting_rates'].values())
            violations = sum(1 for rate in perf['forgetting_rates'].values() if rate > 5.0)
            
            # 優先選擇無違規的配置
            if violations == 0 and total_forgetting < best_total_forgetting:
                best_total_forgetting = total_forgetting
                best_idx = i
            elif best_idx == -1 and total_forgetting < best_total_forgetting:
                best_total_forgetting = total_forgetting
                best_idx = i
                
        if best_idx >= 0:
            return self.weights_history[best_idx]
        return self.current_weights
    
    def get_weight_analysis(self) -> Dict:
        """分析權重變化趨勢"""
        if len(self.weights_history) < 2:
            return {'status': '數據不足', 'trend': {}}
            
        analysis = {
            'current_weights': self.current_weights,
            'weight_trends': {},
            'recommendations': []
        }
        
        # 分析各任務權重趨勢
        for task in self.current_weights:
            weights = [w[task] for w in self.weights_history]
            trend = 'increasing' if weights[-1] > weights[0] else 'decreasing'
            analysis['weight_trends'][task] = {
                'trend': trend,
                'initial': weights[0],
                'current': weights[-1],
                'change': (weights[-1] - weights[0]) / weights[0] * 100
            }
            
        # 提供建議
        latest_perf = self.performance_history[-1] if self.performance_history else None
        if latest_perf:
            for task, rate in latest_perf['forgetting_rates'].items():
                if rate > 5.0:
                    analysis['recommendations'].append(
                        f"{task}: 遺忘率 {rate:.1f}% > 5%，建議繼續增加權重"
                    )
                    
        return analysis
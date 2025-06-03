#!/usr/bin/env python3
"""
依序訓練腳本 - 增強防遺忘版本
實施新的防遺忘策略：學習率調度 + 參數凍結 + 增強正則化
"""
import os
import sys
import argparse
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path

# 添加專案路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.unified_model import create_unified_model
from src.datasets.unified_dataloader import create_unified_dataloaders
from src.utils.sequential_trainer import create_sequential_trainer
from src.utils.ewc import create_ewc_handler


def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(description='依序多任務學習訓練 - 增強防遺忘版本')
    
    # 基本設置
    parser.add_argument('--model_config', type=str, default='default',
                       help='模型配置 (default, small, large)')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='數據目錄路徑')
    parser.add_argument('--save_dir', type=str, default='./sequential_training_enhanced',
                       help='保存目錄路徑')
    
    # 訓練設置
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='數據加載工作線程數')
    parser.add_argument('--device', type=str, default=None,
                       help='計算設備 (cuda/cpu)')
    
    # 階段訓練設置
    parser.add_argument('--stage1_epochs', type=int, default=50,
                       help='階段1 (分割) 訓練輪數')
    parser.add_argument('--stage2_epochs', type=int, default=40,
                       help='階段2 (檢測) 訓練輪數')
    parser.add_argument('--stage3_epochs', type=int, default=30,
                       help='階段3 (分類) 訓練輪數')
    
    # 增強防遺忘策略參數
    parser.add_argument('--ewc_importance', type=float, default=5000.0,
                       help='EWC重要性權重')
    parser.add_argument('--stage1_lr', type=float, default=1e-3,
                       help='階段1學習率')
    parser.add_argument('--stage2_lr', type=float, default=1e-4,
                       help='階段2學習率 (降低10倍)')
    parser.add_argument('--stage3_lr', type=float, default=1e-5,
                       help='階段3學習率 (降低100倍)')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                       help='權重衰減 (增強10倍)')
    parser.add_argument('--freeze_backbone_layers', action='store_true', default=True,
                       help='凍結骨幹網路底層')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help='Dropout率')
    
    # EWC設置
    parser.add_argument('--ewc_type', type=str, default='l2',
                       choices=['l2', 'online'],
                       help='EWC類型')
    parser.add_argument('--adaptive_ewc', action='store_true', default=True,
                       help='使用自適應EWC權重')
    parser.add_argument('--forgetting_threshold', type=float, default=0.05,
                       help='可接受的遺忘率閾值')
    
    # 其他設置
    parser.add_argument('--seed', type=int, default=42,
                       help='隨機種子')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='詳細輸出')
    parser.add_argument('--debug', action='store_true',
                       help='調試模式')
    parser.add_argument('--dry_run', action='store_true',
                       help='空運行')
    
    # 新增選項
    parser.add_argument('--ewc_samples', type=int, default=100,
                       help='計算Fisher矩陣的樣本數')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                       help='梯度裁剪值')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='統一學習率（覆蓋階段學習率）')
    
    args = parser.parse_args()
    
    # 如果提供了統一學習率，覆蓋階段學習率
    if args.learning_rate is not None:
        args.stage1_lr = args.learning_rate
        args.stage2_lr = args.learning_rate / 10  # 階段2降低10倍
        args.stage3_lr = args.learning_rate / 100  # 階段3降低100倍
    
    # 調試模式設置
    if args.debug:
        print("🐛 調試模式啟用")
        args.stage1_epochs = min(args.stage1_epochs, 3)
        args.stage2_epochs = min(args.stage2_epochs, 3)
        args.stage3_epochs = min(args.stage3_epochs, 3)
        args.batch_size = 4
        args.num_workers = 0
    
    return args


def setup_environment(args):
    """設置訓練環境"""
    # 設置隨機種子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 設置設備
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 創建保存目錄
    if not args.dry_run:
        os.makedirs(args.save_dir, exist_ok=True)
        
        # 保存訓練配置
        config_path = os.path.join(args.save_dir, 'training_config.json')
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    
    return args


def freeze_backbone_layers(model, layers_to_freeze=['layer1', 'layer2']):
    """凍結骨幹網路的底層參數"""
    frozen_count = 0
    total_count = 0
    
    for name, param in model.backbone.named_parameters():
        total_count += 1
        if any(layer in name for layer in layers_to_freeze):
            param.requires_grad = False
            frozen_count += 1
    
    print(f"  - 凍結參數: {frozen_count}/{total_count} 個骨幹網路參數")
    
    # 計算剩餘可訓練參數
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - 可訓練參數: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    return model


def add_dropout_to_model(model, dropout_rate=0.3):
    """為模型添加Dropout層增強正則化"""
    # 在頭部網路中添加Dropout
    if hasattr(model.head, 'shared_conv'):
        # 在共享卷積層後添加Dropout
        old_shared = model.head.shared_conv
        model.head.shared_conv = nn.Sequential(
            old_shared,
            nn.Dropout2d(dropout_rate)
        )
    
    print(f"  - 添加Dropout層: rate={dropout_rate}")
    return model


class EnhancedSequentialTrainer:
    """增強的依序訓練器，實施新的防遺忘策略"""
    
    def __init__(self, base_trainer, stage_lrs, weight_decay, gradient_clip):
        self.base_trainer = base_trainer
        self.stage_lrs = stage_lrs  # 階段學習率字典
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        self.current_stage = 0
        
    def train_stage(self, stage_name, task_type, epochs):
        """使用階段特定的學習率訓練"""
        # 獲取當前階段的學習率
        stage_lr = self.stage_lrs.get(self.current_stage, 1e-3)
        print(f"\n  📌 使用學習率: {stage_lr} (階段{self.current_stage + 1})")
        
        # 更新優化器
        self.base_trainer.optimizer = optim.AdamW(
            self.base_trainer.model.parameters(),
            lr=stage_lr,
            weight_decay=self.weight_decay
        )
        
        # 設置學習率調度器
        self.base_trainer.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.base_trainer.optimizer,
            T_max=epochs,
            eta_min=stage_lr * 0.1
        )
        
        # 調用基礎訓練器的訓練方法
        results = self.base_trainer.train_stage(stage_name, task_type, epochs)
        
        # 更新階段計數
        self.current_stage += 1
        
        return results
    
    def __getattr__(self, name):
        """委託給基礎訓練器"""
        return getattr(self.base_trainer, name)


def create_enhanced_trainer(model, dataloaders, ewc, args):
    """創建增強的依序訓練器"""
    # 準備訓練器數據加載器
    trainer_dataloaders = {
        'segmentation_train': dataloaders['train'].get_task_loader('segmentation'),
        'segmentation_val': dataloaders['val'].get_task_loader('segmentation'),
        'detection_train': dataloaders['train'].get_task_loader('detection'),
        'detection_val': dataloaders['val'].get_task_loader('detection'),
        'classification_train': dataloaders['train'].get_task_loader('classification'),
        'classification_val': dataloaders['val'].get_task_loader('classification')
    }
    
    # 創建基礎訓練器
    base_trainer = create_sequential_trainer(
        model=model,
        dataloaders=trainer_dataloaders,
        ewc_importance=args.ewc_importance,
        learning_rate=args.stage1_lr,  # 初始學習率
        device=args.device,
        save_dir=args.save_dir if not args.dry_run else './checkpoints',
        adaptive_ewc=args.adaptive_ewc,
        forgetting_threshold=args.forgetting_threshold
    )
    
    # 使用增強的EWC
    base_trainer.ewc = ewc
    
    # 創建階段學習率字典
    stage_lrs = {
        0: args.stage1_lr,
        1: args.stage2_lr,
        2: args.stage3_lr
    }
    
    # 創建增強訓練器
    enhanced_trainer = EnhancedSequentialTrainer(
        base_trainer=base_trainer,
        stage_lrs=stage_lrs,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip
    )
    
    return enhanced_trainer


def train_with_enhanced_protection(trainer, args):
    """使用增強保護進行訓練"""
    print("\n🚀 開始依序訓練（增強防遺忘保護）")
    print("="*60)
    print("防遺忘策略:")
    print(f"  1. 學習率調度: {args.stage1_lr} → {args.stage2_lr} → {args.stage3_lr}")
    print(f"  2. 參數凍結: {'啟用' if args.freeze_backbone_layers else '禁用'}")
    print(f"  3. 權重衰減: {args.weight_decay}")
    print(f"  4. Dropout: {args.dropout_rate}")
    print(f"  5. EWC權重: {args.ewc_importance}")
    print("="*60)
    
    start_time = time.time()
    results = {}
    
    # 階段1：分割任務
    print(f"\n📌 階段1：分割任務訓練 ({args.stage1_epochs} epochs)")
    stage1_metrics = trainer.train_stage(
        stage_name='stage1_segmentation',
        task_type='segmentation',
        epochs=args.stage1_epochs
    )
    results['stage1'] = stage1_metrics
    
    # 檢查Stage 1性能
    print(f"\n✅ Stage 1 完成:")
    print(f"  - 最佳mIoU: {stage1_metrics.get('best_metric', 0.0):.4f}")
    stage1_time = time.time() - start_time
    print(f"  - 訓練時間: {stage1_time/60:.1f}分鐘")
    
    # 階段2：檢測任務
    print(f"\n📌 階段2：檢測任務訓練 ({args.stage2_epochs} epochs)")
    print(f"  - EWC保護啟動")
    print(f"  - 學習率降低10倍: {args.stage2_lr}")
    
    stage2_metrics = trainer.train_stage(
        stage_name='stage2_detection',
        task_type='detection',
        epochs=args.stage2_epochs
    )
    results['stage2'] = stage2_metrics
    
    # 檢查遺忘情況
    print(f"\n✅ Stage 2 完成:")
    print(f"  - 最佳mAP: {stage2_metrics.get('best_metric', 0.0):.4f}")
    stage2_time = time.time() - start_time - stage1_time
    print(f"  - 訓練時間: {stage2_time/60:.1f}分鐘")
    
    # 計算分割任務的遺忘率
    with torch.no_grad():
        trainer.model.eval()
        current_seg_perf = trainer._validate_segmentation(trainer.dataloaders['segmentation_val'])
    seg_baseline = trainer.baseline_performance.get('segmentation', {}).get('main_metric', 0.0)
    seg_current = current_seg_perf.get('main_metric', current_seg_perf.get('miou', 0.0))
    seg_forgetting = (seg_baseline - seg_current) / seg_baseline if seg_baseline > 0 else 0
    print(f"  - 分割任務遺忘率: {seg_forgetting:.2%} (從 {seg_baseline:.4f} 到 {seg_current:.4f})")
    
    # 階段3：分類任務
    print(f"\n📌 階段3：分類任務訓練 ({args.stage3_epochs} epochs)")
    print(f"  - 完整EWC保護")
    print(f"  - 學習率降低100倍: {args.stage3_lr}")
    
    stage3_metrics = trainer.train_stage(
        stage_name='stage3_classification',
        task_type='classification',
        epochs=args.stage3_epochs
    )
    results['stage3'] = stage3_metrics
    
    # 最終評估
    total_time = time.time() - start_time
    print(f"\n✅ Stage 3 完成:")
    print(f"  - 最佳準確率: {stage3_metrics.get('best_metric', 0.0):.4f}")
    stage3_time = time.time() - start_time - stage1_time - stage2_time
    print(f"  - 訓練時間: {stage3_time/60:.1f}分鐘")
    
    # 計算總遺忘率
    total_forgetting = 0.0
    forgetting_details = {}
    num_tasks = 0
    
    for task in trainer.completed_tasks[:-1]:  # 不包括最後一個任務
        if task in trainer.baseline_performance:
            with torch.no_grad():
                trainer.model.eval()
                if task == 'segmentation':
                    current_perf = trainer._validate_segmentation(trainer.dataloaders[f'{task}_val'])
                elif task == 'detection':
                    current_perf = trainer._validate_detection(trainer.dataloaders[f'{task}_val'])
                elif task == 'classification':
                    current_perf = trainer._validate_classification(trainer.dataloaders[f'{task}_val'])
            
            baseline = trainer.baseline_performance[task].get('main_metric', 0.0)
            current = current_perf.get('main_metric', current_perf.get('miou' if task == 'segmentation' else 'map' if task == 'detection' else 'accuracy', 0.0))
            forgetting = (baseline - current) / baseline if baseline > 0 else 0
            forgetting_details[task] = {
                'baseline': baseline,
                'current': current,
                'forgetting': forgetting
            }
            total_forgetting += forgetting
            num_tasks += 1
    
    total_forgetting = total_forgetting / num_tasks if num_tasks > 0 else 0
    
    # 打印訓練總結
    print("\n" + "="*60)
    print("📊 訓練總結")
    print("="*60)
    print(f"總訓練時間: {total_time/60:.1f} 分鐘")
    print(f"總遺忘率: {total_forgetting:.2%}")
    print(f"最終EWC權重: {trainer.ewc.adaptive_importance:.1f}")
    
    # 各任務最終性能
    print("\n各任務最終性能:")
    final_metrics = {}
    for task in trainer.completed_tasks:
        if f'{task}_val' in trainer.dataloaders:
            with torch.no_grad():
                trainer.model.eval()
                if task == 'segmentation':
                    perf = trainer._validate_segmentation(trainer.dataloaders[f'{task}_val'])
                elif task == 'detection':
                    perf = trainer._validate_detection(trainer.dataloaders[f'{task}_val'])
                elif task == 'classification':
                    perf = trainer._validate_classification(trainer.dataloaders[f'{task}_val'])
            
            metric_value = perf.get('main_metric', 
                                   perf.get('miou' if task == 'segmentation' else 
                                          'map' if task == 'detection' else 
                                          'accuracy', 0.0))
            final_metrics[task] = metric_value
            print(f"  - {task}: {metric_value:.4f}")
    
    # 詳細遺忘分析
    print("\n遺忘率詳情:")
    for task, details in forgetting_details.items():
        print(f"  - {task}: {details['baseline']:.4f} → {details['current']:.4f} (遺忘率: {details['forgetting']:.2%})")
    
    # 檢查是否滿足要求
    if total_forgetting <= args.forgetting_threshold:
        print(f"\n✅ 成功！遺忘率 {total_forgetting:.2%} ≤ {args.forgetting_threshold:.1%}")
    else:
        print(f"\n❌ 失敗！遺忘率 {total_forgetting:.2%} > {args.forgetting_threshold:.1%}")
    
    # 保存最終結果
    if not args.dry_run:
        results['summary'] = {
            'total_time': total_time,
            'total_forgetting': total_forgetting,
            'forgetting_details': forgetting_details,
            'final_ewc_weight': trainer.ewc.adaptive_importance,
            'final_metrics': final_metrics,
            'success': total_forgetting <= args.forgetting_threshold,
            'stage_times': {
                'stage1': stage1_time,
                'stage2': stage2_time,
                'stage3': stage3_time
            },
            'strategy': {
                'stage_lrs': [args.stage1_lr, args.stage2_lr, args.stage3_lr],
                'weight_decay': args.weight_decay,
                'dropout': args.dropout_rate,
                'frozen_layers': args.freeze_backbone_layers,
                'ewc_importance': args.ewc_importance
            }
        }
        
        results_path = os.path.join(args.save_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 結果已保存到: {results_path}")
    
    return results


def main():
    """主函數"""
    # 解析參數
    args = parse_args()
    
    # 設置環境
    args = setup_environment(args)
    
    print("🚀 啟動依序多任務學習訓練（增強防遺忘版本）")
    print("="*60)
    print(f"訓練配置:")
    print(f"  - 模型: {args.model_config}")
    print(f"  - 設備: {args.device}")
    print(f"  - 批次大小: {args.batch_size}")
    print(f"  - 防遺忘策略:")
    print(f"    * 學習率調度: {args.stage1_lr} → {args.stage2_lr} → {args.stage3_lr}")
    print(f"    * 參數凍結: {'是' if args.freeze_backbone_layers else '否'}")
    print(f"    * 權重衰減: {args.weight_decay}")
    print(f"    * Dropout: {args.dropout_rate}")
    print(f"    * EWC權重: {args.ewc_importance}")
    print(f"  - 遺忘率閾值: {args.forgetting_threshold:.1%}")
    print(f"  - 訓練輪數: {args.stage1_epochs}/{args.stage2_epochs}/{args.stage3_epochs}")
    print("="*60)
    
    # 創建模型和數據
    print("\n🔧 創建模型和數據加載器...")
    model = create_unified_model(args.model_config)
    model = model.to(args.device)
    
    # 凍結骨幹網路底層（如果啟用）
    if args.freeze_backbone_layers:
        print("\n🔒 凍結骨幹網路底層參數...")
        model = freeze_backbone_layers(model)
    
    # 添加Dropout增強正則化
    print("\n💧 添加Dropout層...")
    model = add_dropout_to_model(model, args.dropout_rate)
    
    # 統計模型參數
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 模型參數統計:")
    print(f"  - 總參數量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  - 可訓練參數: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # 創建數據加載器
    dataloaders = create_unified_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampling_strategy='balanced',
        task_weights=[1.0, 1.0, 1.0]
    )
    
    # 創建增強的EWC處理器
    print(f"\n🛡️ 創建EWC處理器...")
    ewc = create_ewc_handler(
        model=model,
        importance=args.ewc_importance,
        ewc_type=args.ewc_type
    )
    
    # 創建增強的訓練器
    trainer = create_enhanced_trainer(model, dataloaders, ewc, args)
    
    # 開始訓練
    results = train_with_enhanced_protection(trainer, args)
    
    print("\n✨ 訓練完成！")
    
    # 返回成功狀態
    return 0 if results['summary']['success'] else 1


if __name__ == "__main__":
    exit(main())
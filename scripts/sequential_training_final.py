#!/usr/bin/env python3
"""
最終版依序訓練腳本 - 強化防遺忘策略
採用更激進的參數凍結和學習率策略
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
from pathlib import Path

# 添加專案路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.unified_model import create_unified_model
from src.datasets.unified_dataloader import create_unified_dataloaders
from src.utils.sequential_trainer import create_sequential_trainer
from src.utils.ewc import create_ewc_handler


def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(description='最終版依序多任務學習訓練')
    
    # 基本設置
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./final_results')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # 訓練輪數
    parser.add_argument('--stage1_epochs', type=int, default=30)
    parser.add_argument('--stage2_epochs', type=int, default=20)
    parser.add_argument('--stage3_epochs', type=int, default=15)
    
    # 防遺忘策略參數
    parser.add_argument('--ewc_importance', type=float, default=10000.0)
    parser.add_argument('--stage1_lr', type=float, default=1e-3)
    parser.add_argument('--stage2_lr', type=float, default=1e-5)  # 極低學習率
    parser.add_argument('--stage3_lr', type=float, default=5e-6)  # 更低學習率
    parser.add_argument('--weight_decay', type=float, default=5e-3)  # 強正則化
    
    # 其他參數
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()
    
    if args.debug:
        args.stage1_epochs = 5
        args.stage2_epochs = 3
        args.stage3_epochs = 2
        args.batch_size = 8
    
    return args


def aggressive_freeze_strategy(model, stage):
    """激進的參數凍結策略"""
    frozen_params = 0
    total_params = 0
    
    if stage == 1:
        # Stage 1: 不凍結任何參數
        return model
    
    elif stage == 2:
        # Stage 2: 凍結骨幹網路的前60%層
        backbone_params = list(model.backbone.named_parameters())
        freeze_count = int(len(backbone_params) * 0.6)
        
        for idx, (name, param) in enumerate(backbone_params):
            total_params += 1
            if idx < freeze_count:
                param.requires_grad = False
                frozen_params += 1
        
        # 同時凍結分割頭部
        for name, param in model.head.named_parameters():
            if 'segmentation' in name:
                param.requires_grad = False
                frozen_params += 1
            total_params += 1
    
    elif stage == 3:
        # Stage 3: 凍結所有骨幹網路和已訓練的任務頭部
        # 凍結整個骨幹網路
        for name, param in model.backbone.named_parameters():
            param.requires_grad = False
            frozen_params += 1
            total_params += 1
        
        # 凍結共享層
        for name, param in model.head.named_parameters():
            if 'shared' in name:
                param.requires_grad = False
                frozen_params += 1
            total_params += 1
        
        # 凍結分割和檢測頭部
        for name, param in model.head.named_parameters():
            if 'segmentation' in name or 'detection' in name:
                param.requires_grad = False
                frozen_params += 1
    
    # 統計可訓練參數
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_model_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n🔒 階段 {stage} 參數凍結策略:")
    print(f"  - 凍結參數數量: {frozen_params}")
    print(f"  - 可訓練參數: {trainable_params:,} / {total_model_params:,}")
    print(f"  - 可訓練比例: {trainable_params/total_model_params*100:.1f}%")
    
    return model


def create_stage_optimizer(model, stage, base_lr, weight_decay):
    """為每個階段創建特定的優化器"""
    # 只優化未凍結的參數
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    
    if stage == 1:
        # Stage 1: 標準優化器
        optimizer = optim.AdamW(params_to_optimize, lr=base_lr, weight_decay=weight_decay)
    elif stage == 2:
        # Stage 2: 更低的學習率，更高的動量
        optimizer = optim.SGD(params_to_optimize, lr=base_lr, 
                            momentum=0.95, weight_decay=weight_decay, nesterov=True)
    else:
        # Stage 3: 極低學習率的SGD
        optimizer = optim.SGD(params_to_optimize, lr=base_lr, 
                            momentum=0.99, weight_decay=weight_decay)
    
    return optimizer


def main():
    args = parse_args()
    
    # 設置環境
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("🚀 最終版依序多任務學習訓練")
    print("="*60)
    print("強化防遺忘策略:")
    print(f"  1. 激進參數凍結")
    print(f"  2. 極低學習率: {args.stage1_lr} → {args.stage2_lr} → {args.stage3_lr}")
    print(f"  3. 強正則化: weight_decay={args.weight_decay}")
    print(f"  4. 高EWC權重: {args.ewc_importance}")
    print("="*60)
    
    # 創建模型
    print("\n📦 創建模型...")
    model = create_unified_model('default')
    model = model.to(args.device)
    
    # 創建數據加載器
    print("\n📊 創建數據加載器...")
    dataloaders = create_unified_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampling_strategy='balanced'
    )
    
    # 準備訓練器數據
    trainer_dataloaders = {
        'segmentation_train': dataloaders['train'].get_task_loader('segmentation'),
        'segmentation_val': dataloaders['val'].get_task_loader('segmentation'),
        'detection_train': dataloaders['train'].get_task_loader('detection'),
        'detection_val': dataloaders['val'].get_task_loader('detection'),
        'classification_train': dataloaders['train'].get_task_loader('classification'),
        'classification_val': dataloaders['val'].get_task_loader('classification')
    }
    
    # 創建EWC處理器
    print(f"\n🛡️ 創建EWC處理器 (importance={args.ewc_importance})...")
    ewc = create_ewc_handler(
        model=model,
        importance=args.ewc_importance,
        ewc_type='l2'
    )
    
    # 記錄結果
    results = {
        'config': vars(args),
        'stages': {},
        'forgetting_rates': {}
    }
    
    # Stage 1: 分割任務
    print(f"\n📌 階段1：分割任務 ({args.stage1_epochs} epochs)")
    print(f"  學習率: {args.stage1_lr}")
    
    # 不凍結任何參數
    model = aggressive_freeze_strategy(model, stage=1)
    
    # 創建優化器
    optimizer = create_stage_optimizer(model, 1, args.stage1_lr, args.weight_decay)
    
    # 創建訓練器
    trainer = create_sequential_trainer(
        model=model,
        dataloaders=trainer_dataloaders,
        ewc_importance=args.ewc_importance,
        learning_rate=args.stage1_lr,
        device=args.device,
        save_dir=args.save_dir,
        adaptive_ewc=False,  # 關閉自適應，使用固定權重
        forgetting_threshold=0.05
    )
    
    # 使用我們的EWC和優化器
    trainer.ewc = ewc
    trainer.optimizer = optimizer
    
    # 訓練Stage 1
    stage1_start = time.time()
    stage1_metrics = trainer.train_stage(
        stage_name='stage1_segmentation',
        task_type='segmentation',
        epochs=args.stage1_epochs
    )
    stage1_time = time.time() - stage1_start
    
    results['stages']['segmentation'] = {
        'metrics': stage1_metrics,
        'time': stage1_time,
        'best_score': stage1_metrics.get('best_metric', 0)
    }
    
    print(f"\n✅ Stage 1 完成!")
    print(f"  最佳mIoU: {stage1_metrics.get('best_metric', 0):.4f}")
    print(f"  訓練時間: {stage1_time/60:.1f}分鐘")
    
    # 保存Stage 1的基準性能
    baseline_seg = stage1_metrics.get('best_metric', 0)
    
    # Stage 2: 檢測任務
    print(f"\n📌 階段2：檢測任務 ({args.stage2_epochs} epochs)")
    print(f"  學習率: {args.stage2_lr} (降低{args.stage1_lr/args.stage2_lr:.0f}倍)")
    
    # 激進凍結策略
    model = aggressive_freeze_strategy(model, stage=2)
    
    # 創建新優化器（只優化未凍結參數）
    optimizer = create_stage_optimizer(model, 2, args.stage2_lr, args.weight_decay)
    trainer.optimizer = optimizer
    
    # 訓練Stage 2
    stage2_start = time.time()
    stage2_metrics = trainer.train_stage(
        stage_name='stage2_detection',
        task_type='detection',
        epochs=args.stage2_epochs
    )
    stage2_time = time.time() - stage2_start
    
    results['stages']['detection'] = {
        'metrics': stage2_metrics,
        'time': stage2_time,
        'best_score': stage2_metrics.get('best_metric', 0)
    }
    
    # 檢查分割任務遺忘
    with torch.no_grad():
        model.eval()
        seg_perf = trainer._validate_segmentation(trainer_dataloaders['segmentation_val'])
        current_seg = seg_perf.get('main_metric', seg_perf.get('miou', 0))
    
    seg_forgetting = (baseline_seg - current_seg) / baseline_seg if baseline_seg > 0 else 0
    results['forgetting_rates']['segmentation_after_detection'] = seg_forgetting
    
    print(f"\n✅ Stage 2 完成!")
    print(f"  最佳mAP: {stage2_metrics.get('best_metric', 0):.4f}")
    print(f"  分割遺忘率: {seg_forgetting:.2%}")
    print(f"  訓練時間: {stage2_time/60:.1f}分鐘")
    
    # Stage 3: 分類任務
    print(f"\n📌 階段3：分類任務 ({args.stage3_epochs} epochs)")
    print(f"  學習率: {args.stage3_lr} (降低{args.stage1_lr/args.stage3_lr:.0f}倍)")
    
    # 最激進的凍結策略
    model = aggressive_freeze_strategy(model, stage=3)
    
    # 創建新優化器
    optimizer = create_stage_optimizer(model, 3, args.stage3_lr, args.weight_decay)
    trainer.optimizer = optimizer
    
    # 訓練Stage 3
    stage3_start = time.time()
    stage3_metrics = trainer.train_stage(
        stage_name='stage3_classification',
        task_type='classification',
        epochs=args.stage3_epochs
    )
    stage3_time = time.time() - stage3_start
    
    results['stages']['classification'] = {
        'metrics': stage3_metrics,
        'time': stage3_time,
        'best_score': stage3_metrics.get('best_metric', 0)
    }
    
    # 最終評估所有任務
    print("\n📊 最終性能評估...")
    final_performance = {}
    
    with torch.no_grad():
        model.eval()
        
        # 分割
        seg_perf = trainer._validate_segmentation(trainer_dataloaders['segmentation_val'])
        final_seg = seg_perf.get('main_metric', seg_perf.get('miou', 0))
        final_performance['segmentation'] = final_seg
        
        # 檢測
        det_perf = trainer._validate_detection(trainer_dataloaders['detection_val'])
        final_det = det_perf.get('main_metric', det_perf.get('map', 0))
        final_performance['detection'] = final_det
        
        # 分類
        cls_perf = trainer._validate_classification(trainer_dataloaders['classification_val'])
        final_cls = cls_perf.get('main_metric', cls_perf.get('accuracy', 0))
        final_performance['classification'] = final_cls
    
    # 計算最終遺忘率
    seg_forgetting_final = (baseline_seg - final_seg) / baseline_seg if baseline_seg > 0 else 0
    det_forgetting_final = (stage2_metrics.get('best_metric', 0) - final_det) / stage2_metrics.get('best_metric', 1) if stage2_metrics.get('best_metric', 0) > 0 else 0
    
    avg_forgetting = (seg_forgetting_final + det_forgetting_final) / 2
    
    results['final_performance'] = final_performance
    results['final_forgetting_rates'] = {
        'segmentation': seg_forgetting_final,
        'detection': det_forgetting_final,
        'average': avg_forgetting
    }
    results['total_time'] = stage1_time + stage2_time + stage3_time
    
    # 打印總結
    print("\n" + "="*60)
    print("🏆 訓練完成總結")
    print("="*60)
    print(f"總訓練時間: {results['total_time']/60:.1f}分鐘")
    print("\n最終性能:")
    print(f"  分割 mIoU: {final_seg:.4f} (遺忘率: {seg_forgetting_final:.2%})")
    print(f"  檢測 mAP: {final_det:.4f} (遺忘率: {det_forgetting_final:.2%})")
    print(f"  分類準確率: {final_cls:.4f}")
    print(f"\n平均遺忘率: {avg_forgetting:.2%}")
    
    if avg_forgetting <= 0.05:
        print("\n✅ 成功！平均遺忘率 ≤ 5%")
    else:
        print(f"\n⚠️ 平均遺忘率 {avg_forgetting:.2%} > 5%")
    
    # 保存結果
    results_path = os.path.join(args.save_dir, 'final_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 結果已保存到: {results_path}")
    
    return 0 if avg_forgetting <= 0.05 else 1


if __name__ == "__main__":
    exit(main())
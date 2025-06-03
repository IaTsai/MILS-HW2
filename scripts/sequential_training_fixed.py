#!/usr/bin/env python3
"""
依序訓練腳本 - 修復版本
使用增強的EWC防遺忘策略，確保遺忘率控制在5%以內
"""
import os
import sys
import argparse
import json
import time
import torch
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
    parser = argparse.ArgumentParser(description='依序多任務學習訓練 - 修復版本')
    
    # 基本設置
    parser.add_argument('--model_config', type=str, default='default',
                       help='模型配置 (default, small, large)')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='數據目錄路徑')
    parser.add_argument('--save_dir', type=str, default='./sequential_training_results_fixed',
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
    
    # EWC設置 - 使用更高的預設值
    parser.add_argument('--ewc_importance', type=float, default=50000.0,
                       help='EWC重要性權重 (大幅提高到50000)')
    parser.add_argument('--ewc_type', type=str, default='l2',
                       choices=['l2', 'online'],
                       help='EWC類型')
    parser.add_argument('--adaptive_ewc', action='store_true', default=True,
                       help='使用自適應EWC權重 (預設啟用)')
    parser.add_argument('--forgetting_threshold', type=float, default=0.03,
                       help='可接受的遺忘率閾值 (更嚴格的3%)')
    
    # 優化器設置
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='初始學習率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='權重衰減')
    
    # 其他設置
    parser.add_argument('--seed', type=int, default=42,
                       help='隨機種子')
    parser.add_argument('--resume', type=str, default=None,
                       help='從檢查點恢復訓練')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='詳細輸出')
    parser.add_argument('--debug', action='store_true',
                       help='調試模式 (減少訓練輪數)')
    parser.add_argument('--dry_run', action='store_true',
                       help='空運行 (不保存結果)')
    
    # 新增選項
    parser.add_argument('--ewc_samples', type=int, default=100,
                       help='計算Fisher矩陣的樣本數')
    parser.add_argument('--max_fisher_value', type=float, default=1e6,
                       help='Fisher值上限')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                       help='梯度裁剪值')
    
    args = parser.parse_args()
    
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


def create_models_and_dataloaders(args):
    """創建模型和數據加載器"""
    print("\n🔧 創建模型和數據加載器...")
    
    # 創建統一模型
    model = create_unified_model(args.model_config)
    model = model.to(args.device)
    
    # 統計模型參數
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  📊 總參數量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  📊 可訓練參數: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # 創建數據加載器
    dataloaders = create_unified_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampling_strategy='balanced',
        task_weights=[1.0, 1.0, 1.0]
    )
    
    # 統計數據集大小
    # UnifiedDataLoader使用unified_dataset屬性
    train_size = len(dataloaders['train'].unified_dataset)
    val_size = len(dataloaders['val'].unified_dataset)
    print(f"  📊 訓練集大小: {train_size}")
    print(f"  📊 驗證集大小: {val_size}")
    
    return model, dataloaders


def create_enhanced_ewc_handler(model, args):
    """創建增強的EWC處理器"""
    print(f"\n🛡️ 創建增強EWC處理器...")
    print(f"  - 初始權重: {args.ewc_importance}")
    print(f"  - EWC類型: {args.ewc_type}")
    print(f"  - 自適應調整: {'啟用' if args.adaptive_ewc else '禁用'}")
    print(f"  - Fisher值上限: {args.max_fisher_value}")
    print(f"  - 遺忘率閾值: {args.forgetting_threshold:.1%}")
    
    ewc = create_ewc_handler(
        model=model,
        importance=args.ewc_importance,
        ewc_type=args.ewc_type,
        max_fisher_value=args.max_fisher_value
    )
    
    return ewc


def create_enhanced_trainer(model, dataloaders, ewc, args):
    """創建增強的依序訓練器"""
    # 準備訓練器數據加載器 - 使用SequentialTrainer期望的鍵名格式
    trainer_dataloaders = {
        'segmentation_train': dataloaders['train'].get_task_loader('segmentation'),
        'segmentation_val': dataloaders['val'].get_task_loader('segmentation'),
        'detection_train': dataloaders['train'].get_task_loader('detection'),
        'detection_val': dataloaders['val'].get_task_loader('detection'),
        'classification_train': dataloaders['train'].get_task_loader('classification'),
        'classification_val': dataloaders['val'].get_task_loader('classification')
    }
    
    # 創建訓練器配置 - 只包含SequentialTrainer支持的參數
    trainer_config = {
        'learning_rate': args.learning_rate,
        'device': args.device,
        'save_dir': args.save_dir if not args.dry_run else './checkpoints',
        'adaptive_ewc': args.adaptive_ewc,
        'forgetting_threshold': args.forgetting_threshold
    }
    
    # 創建依序訓練器
    trainer = create_sequential_trainer(
        model=model,
        dataloaders=trainer_dataloaders,
        ewc_importance=args.ewc_importance,
        **trainer_config
    )
    
    # 使用我們的增強EWC替換訓練器的EWC
    trainer.ewc = ewc
    
    return trainer


def train_with_enhanced_protection(trainer, args):
    """使用增強保護進行訓練"""
    print("\n🚀 開始依序訓練（增強防遺忘保護）")
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
    print(f"  - 最佳mIoU: {stage1_metrics.get('main_metric', stage1_metrics.get('miou', 0.0)):.4f}")
    stage1_time = time.time() - start_time
    print(f"  - 訓練時間: {stage1_time:.1f}秒")
    
    # 階段2：檢測任務（帶EWC保護）
    print(f"\n📌 階段2：檢測任務訓練 ({args.stage2_epochs} epochs) - EWC保護啟動")
    print(f"  - 當前EWC權重: {trainer.ewc.adaptive_importance:.1f}")
    
    stage2_metrics = trainer.train_stage(
        stage_name='stage2_detection',
        task_type='detection',
        epochs=args.stage2_epochs
    )
    results['stage2'] = stage2_metrics
    
    # 檢查遺忘情況
    print(f"\n✅ Stage 2 完成:")
    print(f"  - 最佳mAP: {stage2_metrics.get('main_metric', stage2_metrics.get('map', 0.0)):.4f}")
    stage2_time = time.time() - start_time - stage1_time
    print(f"  - 訓練時間: {stage2_time:.1f}秒")
    # 計算分割任務的遺忘率
    with torch.no_grad():
        trainer.model.eval()
        current_seg_perf = trainer._validate_segmentation(trainer.dataloaders['segmentation_val'])
    seg_baseline = trainer.baseline_performance.get('segmentation', {}).get('main_metric', 0.0)
    seg_current = current_seg_perf.get('main_metric', 0.0)
    seg_forgetting = (seg_baseline - seg_current) / seg_baseline if seg_baseline > 0 else 0
    print(f"  - 分割任務遺忘率: {seg_forgetting:.2%} (從 {seg_baseline:.4f} 到 {seg_current:.4f})")
    
    if seg_forgetting > args.forgetting_threshold:
        print(f"  ⚠️ 遺忘率超標！調整後的EWC權重: {trainer.ewc.adaptive_importance:.1f}")
    
    # 階段3：分類任務（帶EWC保護）
    print(f"\n📌 階段3：分類任務訓練 ({args.stage3_epochs} epochs) - 完整EWC保護")
    print(f"  - 當前EWC權重: {trainer.ewc.adaptive_importance:.1f}")
    
    stage3_metrics = trainer.train_stage(
        stage_name='stage3_classification',
        task_type='classification',
        epochs=args.stage3_epochs
    )
    results['stage3'] = stage3_metrics
    
    # 最終評估
    total_time = time.time() - start_time
    print(f"\n✅ Stage 3 完成:")
    print(f"  - 最佳準確率: {stage3_metrics.get('main_metric', stage3_metrics.get('accuracy', 0.0)):.4f}")
    stage3_time = time.time() - start_time - stage1_time - stage2_time
    print(f"  - 訓練時間: {stage3_time:.1f}秒")
    
    # 計算總遺忘率
    total_forgetting = 0.0
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
            current = current_perf.get('main_metric', 0.0)
            forgetting = (baseline - current) / baseline if baseline > 0 else 0
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
            final_metrics[task] = perf.get('main_metric', 0.0)
            print(f"  - {task}: {final_metrics[task]:.4f}")
    
    # 檢查是否滿足要求
    if total_forgetting <= args.forgetting_threshold:
        print(f"\n✅ 成功！遺忘率 {total_forgetting:.2%} ≤ {args.forgetting_threshold:.1%}")
    else:
        print(f"\n❌ 失敗！遺忘率 {total_forgetting:.2%} > {args.forgetting_threshold:.1%}")
        print(f"\n建議:")
        print(f"  1. 使用更高的EWC權重: {args.ewc_importance * 2}")
        print(f"  2. 啟用Online EWC減少計算開銷")
        print(f"  3. 增加Fisher矩陣計算樣本數")
    
    # 保存最終結果
    if not args.dry_run:
        results['summary'] = {
            'total_time': total_time,
            'total_forgetting': total_forgetting,
            'final_ewc_weight': trainer.ewc.adaptive_importance,
            'final_metrics': final_metrics,
            'success': total_forgetting <= args.forgetting_threshold,
            'stage_times': {
                'stage1': stage1_time,
                'stage2': stage2_time,
                'stage3': stage3_time
            }
        }
        
        results_path = os.path.join(args.save_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 結果已保存到: {results_path}")
    
    return results


def visualize_results(results, args):
    """可視化訓練結果"""
    if args.dry_run:
        return
    
    print("\n📊 生成訓練曲線...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 從訓練歷史提取數據
    history_path = os.path.join(args.save_dir, 'training_history.json')
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        # 繪製損失曲線
        ax = axes[0, 0]
        for stage in ['stage1', 'stage2', 'stage3']:
            if stage in history:
                epochs = range(1, len(history[stage]['loss']) + 1)
                ax.plot(epochs, history[stage]['loss'], label=f'{stage} loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True)
        
        # 繪製指標曲線
        ax = axes[0, 1]
        metrics_map = {
            'stage1': ('mIoU', 'mIoU'),
            'stage2': ('mAP', 'mAP'),
            'stage3': ('accuracy', 'Accuracy')
        }
        
        for stage, (metric_key, label) in metrics_map.items():
            if stage in history and metric_key in history[stage]:
                epochs = range(1, len(history[stage][metric_key]) + 1)
                ax.plot(epochs, history[stage][metric_key], label=f'{stage} {label}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Metric')
        ax.set_title('Validation Metrics')
        ax.legend()
        ax.grid(True)
        
        # 繪製EWC權重變化
        ax = axes[1, 0]
        if 'ewc_importance' in history:
            epochs = range(1, len(history['ewc_importance']) + 1)
            ax.plot(epochs, history['ewc_importance'], 'r-', linewidth=2)
        ax.set_xlabel('Training Progress')
        ax.set_ylabel('EWC Weight')
        ax.set_title('Adaptive EWC Weight')
        ax.grid(True)
        
        # 繪製遺忘率
        ax = axes[1, 1]
        if 'forgetting_rates' in history:
            stages = list(history['forgetting_rates'].keys())
            rates = list(history['forgetting_rates'].values())
            colors = ['green' if r <= args.forgetting_threshold else 'red' for r in rates]
            bars = ax.bar(stages, rates, color=colors)
            ax.axhline(y=args.forgetting_threshold, color='black', linestyle='--', 
                      label=f'Threshold ({args.forgetting_threshold:.1%})')
            ax.set_ylabel('Forgetting Rate')
            ax.set_title('Task Forgetting Rates')
            ax.legend()
            
            # 添加數值標籤
            for bar, rate in zip(bars, rates):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{rate:.1%}', ha='center', va='bottom')
    
    plt.tight_layout()
    save_path = os.path.join(args.save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150)
    print(f"  📊 訓練曲線已保存到: {save_path}")
    plt.close()


def main():
    """主函數"""
    # 解析參數
    args = parse_args()
    
    # 設置環境
    args = setup_environment(args)
    
    print("🚀 啟動依序多任務學習訓練（修復版本）")
    print("="*60)
    print(f"訓練配置:")
    print(f"  - 模型: {args.model_config}")
    print(f"  - 設備: {args.device}")
    print(f"  - 批次大小: {args.batch_size}")
    print(f"  - EWC權重: {args.ewc_importance}")
    print(f"  - 自適應EWC: {'是' if args.adaptive_ewc else '否'}")
    print(f"  - 遺忘率閾值: {args.forgetting_threshold:.1%}")
    print(f"  - 訓練輪數: {args.stage1_epochs}/{args.stage2_epochs}/{args.stage3_epochs}")
    print("="*60)
    
    # 創建模型和數據
    model, dataloaders = create_models_and_dataloaders(args)
    
    # 創建增強的EWC處理器
    ewc = create_enhanced_ewc_handler(model, args)
    
    # 創建增強的訓練器
    trainer = create_enhanced_trainer(model, dataloaders, ewc, args)
    
    # 開始訓練
    results = train_with_enhanced_protection(trainer, args)
    
    # 可視化結果
    visualize_results(results, args)
    
    print("\n✨ 訓練完成！")
    
    # 返回成功狀態
    return 0 if results['summary']['success'] else 1


if __name__ == "__main__":
    exit(main())
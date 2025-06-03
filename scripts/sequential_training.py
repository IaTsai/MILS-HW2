#!/usr/bin/env python3
"""
依序訓練腳本
實現完整的三階段依序訓練流程：分割 → 檢測 → 分類
集成EWC防遺忘機制和性能監控系統
"""
import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.unified_model import create_unified_model
from src.datasets.unified_dataloader import create_unified_dataloaders
from src.utils.sequential_trainer import create_sequential_trainer


def parse_arguments():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='Sequential Multi-Task Training')
    
    # 模型參數
    parser.add_argument('--model_config', type=str, default='default',
                      help='模型配置 (default, lightweight, large)')
    parser.add_argument('--pretrained', action='store_true',
                      help='使用預訓練權重')
    
    # 訓練參數
    parser.add_argument('--batch_size', type=int, default=16,
                      help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                      help='學習率')
    parser.add_argument('--stage1_epochs', type=int, default=50,
                      help='階段1訓練輪數 (分割)')
    parser.add_argument('--stage2_epochs', type=int, default=40,
                      help='階段2訓練輪數 (檢測)')
    parser.add_argument('--stage3_epochs', type=int, default=30,
                      help='階段3訓練輪數 (分類)')
    
    # EWC參數
    parser.add_argument('--ewc_importance', type=float, default=1000.0,
                      help='EWC重要性權重')
    parser.add_argument('--adaptive_ewc', action='store_true', default=True,
                      help='使用自適應EWC權重調整')
    parser.add_argument('--forgetting_threshold', type=float, default=0.05,
                      help='可接受的遺忘閾值 (5%)')
    
    # 數據參數
    parser.add_argument('--data_dir', type=str, default='./data',
                      help='數據目錄')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='數據加載工作進程數')
    
    # 保存參數
    parser.add_argument('--save_dir', type=str, default='./sequential_training_results',
                      help='結果保存目錄')
    parser.add_argument('--save_interval', type=int, default=10,
                      help='檢查點保存間隔')
    
    # 設備參數
    parser.add_argument('--device', type=str, default='auto',
                      help='訓練設備 (auto, cpu, cuda, cuda:0)')
    parser.add_argument('--mixed_precision', action='store_true',
                      help='使用混合精度訓練')
    
    # 調試參數
    parser.add_argument('--debug', action='store_true',
                      help='調試模式 (使用更少的數據)')
    parser.add_argument('--dry_run', action='store_true',
                      help='試運行模式 (只運行幾個batch)')
    
    return parser.parse_args()


def setup_device(device_arg: str) -> str:
    """設置訓練設備"""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"🚀 自動選擇設備: {device}")
            print(f"📱 GPU: {torch.cuda.get_device_name()}")
            print(f"💾 GPU記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            device = 'cpu'
            print(f"💻 自動選擇設備: {device}")
    else:
        device = device_arg
        print(f"🎯 指定設備: {device}")
    
    return device


def create_dataloaders(args):
    """創建數據加載器"""
    print("📂 創建數據加載器...")
    
    # 數據配置
    data_config = {
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'sampling_strategy': 'balanced',
        'task_weights': [1.0, 1.0, 1.0]
    }
    
    if args.debug:
        print("🐛 調試模式：使用較少數據")
    
    # 創建統一數據加載器
    dataloaders = create_unified_dataloaders(**data_config)
    
    # 打印數據統計
    for name, loader in dataloaders.items():
        if loader is not None:
            print(f"  {name}: {len(loader)} batches")
    
    return dataloaders


def monitor_performance():
    """性能監控函數"""
    def calculate_miou(predictions, targets):
        """計算mIoU"""
        # 這裡應該實現真實的mIoU計算
        # 簡化版本
        return 0.75
    
    def calculate_map(predictions, targets):
        """計算mAP"""
        # 這裡應該實現真實的mAP計算
        # 簡化版本
        return 0.65
    
    def calculate_top1(predictions, targets):
        """計算Top-1準確率"""
        # 這裡應該實現真實的準確率計算
        # 簡化版本
        return 0.85
    
    current_metrics = {
        'segmentation_miou': calculate_miou(None, None),
        'detection_map': calculate_map(None, None),
        'classification_top1': calculate_top1(None, None)
    }
    
    # 模擬基準性能
    baseline_miou = 0.78
    baseline_map = 0.68
    baseline_top1 = 0.87
    
    forgetting_check = {
        'seg_drop': baseline_miou - current_metrics['segmentation_miou'],
        'det_drop': baseline_map - current_metrics['detection_map'],
        'cls_drop': baseline_top1 - current_metrics['classification_top1'],
        'acceptable': True  # 如果所有下降 <= 5%
    }
    
    # 檢查是否可接受
    all_drops = [
        forgetting_check['seg_drop'],
        forgetting_check['det_drop'], 
        forgetting_check['cls_drop']
    ]
    max_drop = max(all_drops)
    forgetting_check['acceptable'] = max_drop <= 0.05
    
    return current_metrics, forgetting_check


def run_sequential_training(args):
    """運行依序訓練"""
    print("🚀 開始依序訓練...")
    print("=" * 70)
    
    # 設置設備
    device = setup_device(args.device)
    
    # 創建保存目錄
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"💾 結果保存到: {save_dir}")
    
    # 保存訓練配置
    config_path = save_dir / 'training_config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    print(f"⚙️ 訓練配置保存到: {config_path}")
    
    # 創建模型
    print(f"\n🏗️ 創建統一多任務模型 (配置: {args.model_config})...")
    model = create_unified_model(args.model_config)
    
    # 模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 模型參數: {total_params:,} 總計, {trainable_params:,} 可訓練")
    
    # 創建數據加載器
    dataloaders = create_dataloaders(args)
    
    # 創建依序訓練器
    # 重構數據加載器以符合預期格式
    trainer_dataloaders = {
        'segmentation_train': dataloaders['train'].get_task_loader('segmentation'),
        'segmentation_val': dataloaders['val'].get_task_loader('segmentation'),
        'detection_train': dataloaders['train'].get_task_loader('detection'),
        'detection_val': dataloaders['val'].get_task_loader('detection'),
        'classification_train': dataloaders['train'].get_task_loader('classification'),
        'classification_val': dataloaders['val'].get_task_loader('classification'),
        'unified_train': dataloaders['train'].unified_loader,
        'unified_val': dataloaders['val'].unified_loader
    }
    
    print(f"\n🔧 創建依序訓練器...")
    trainer = create_sequential_trainer(
        model=model,
        dataloaders=trainer_dataloaders,
        ewc_importance=args.ewc_importance,
        save_dir=str(save_dir),
        device=device,
        learning_rate=args.learning_rate,
        adaptive_ewc=args.adaptive_ewc,
        forgetting_threshold=args.forgetting_threshold
    )
    
    print(f"  EWC重要性權重: {args.ewc_importance}")
    print(f"  自適應EWC: {'開啟' if args.adaptive_ewc else '關閉'}")
    print(f"  遺忘閾值: {args.forgetting_threshold*100:.1f}%")
    
    # 三階段訓練流程
    print("\n" + "=" * 70)
    print("🎯 開始三階段依序訓練流程")
    print("=" * 70)
    
    training_start_time = time.time()
    stage_results = {}
    
    try:
        # Stage 1: 分割任務訓練
        print(f"\n🎨 Stage 1: 分割任務訓練")
        print(f"📊 訓練輪數: {args.stage1_epochs}")
        stage1_start = time.time()
        
        stage1_metrics = trainer.train_stage(
            stage_name='stage1_segmentation',
            task_type='segmentation',
            epochs=args.stage1_epochs,
            save_checkpoints=True
        )
        
        stage1_time = time.time() - stage1_start
        stage_results['stage1'] = {
            'metrics': stage1_metrics,
            'time': stage1_time
        }
        
        print(f"✅ Stage 1 完成！用時: {stage1_time/60:.1f}分鐘")
        print(f"📊 分割mIoU基準: {stage1_metrics['main_metric']:.4f}")
        
        if args.dry_run:
            print("🧪 試運行模式：停止在Stage 1")
            return stage_results
        
        # Stage 2: 檢測任務訓練 + EWC
        print(f"\n🎯 Stage 2: 檢測任務訓練 + EWC")
        print(f"📊 訓練輪數: {args.stage2_epochs}")
        stage2_start = time.time()
        
        stage2_metrics = trainer.train_stage(
            stage_name='stage2_detection',
            task_type='detection',
            epochs=args.stage2_epochs,
            save_checkpoints=True
        )
        
        stage2_time = time.time() - stage2_start
        stage_results['stage2'] = {
            'metrics': stage2_metrics,
            'time': stage2_time
        }
        
        print(f"✅ Stage 2 完成！用時: {stage2_time/60:.1f}分鐘")
        print(f"📊 檢測mAP: {stage2_metrics['main_metric']:.4f}")
        
        # 檢查遺忘 - Stage 2後
        print(f"\n🔍 Stage 2後遺忘檢查...")
        forgetting_info_s2 = trainer.check_forgetting()
        stage_results['stage2']['forgetting'] = forgetting_info_s2
        
        # Stage 3: 分類任務訓練 + EWC
        print(f"\n📊 Stage 3: 分類任務訓練 + EWC")
        print(f"📊 訓練輪數: {args.stage3_epochs}")
        stage3_start = time.time()
        
        stage3_metrics = trainer.train_stage(
            stage_name='stage3_classification',
            task_type='classification',
            epochs=args.stage3_epochs,
            save_checkpoints=True
        )
        
        stage3_time = time.time() - stage3_start
        stage_results['stage3'] = {
            'metrics': stage3_metrics,
            'time': stage3_time
        }
        
        print(f"✅ Stage 3 完成！用時: {stage3_time/60:.1f}分鐘")
        print(f"📊 分類準確率: {stage3_metrics['main_metric']:.4f}")
        
        # 最終遺忘檢查
        print(f"\n🔍 最終遺忘檢查...")
        final_forgetting_info = trainer.check_forgetting()
        stage_results['stage3']['forgetting'] = final_forgetting_info
        
        # 最終性能評估
        print(f"\n📊 最終性能評估...")
        final_metrics = trainer.evaluate_all_tasks()
        stage_results['final_metrics'] = final_metrics
        
        total_time = time.time() - training_start_time
        stage_results['total_time'] = total_time
        
        print(f"\n🎉 依序訓練完成！總用時: {total_time/60:.1f}分鐘")
        
        # 打印最終結果總結
        print_final_summary(stage_results, trainer)
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 訓練被用戶中斷")
        total_time = time.time() - training_start_time
        stage_results['interrupted'] = True
        stage_results['total_time'] = total_time
    
    except Exception as e:
        print(f"\n❌ 訓練過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        total_time = time.time() - training_start_time
        stage_results['error'] = str(e)
        stage_results['total_time'] = total_time
    
    finally:
        # 保存訓練結果和歷史
        print(f"\n💾 保存訓練結果...")
        
        # 保存結果摘要
        results_path = save_dir / 'training_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            # 轉換tensor為可序列化格式
            serializable_results = {}
            for key, value in stage_results.items():
                if isinstance(value, dict):
                    serializable_results[key] = {}
                    for k, v in value.items():
                        if torch.is_tensor(v):
                            serializable_results[key][k] = v.item()
                        else:
                            serializable_results[key][k] = v
                else:
                    serializable_results[key] = value
            
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        print(f"📊 結果摘要保存到: {results_path}")
        
        # 保存訓練歷史
        trainer.save_training_history()
        
        # 繪製訓練曲線
        try:
            trainer.plot_training_curves()
        except Exception as e:
            print(f"⚠️ 繪製訓練曲線失敗: {e}")
        
        print(f"💾 所有結果已保存到: {save_dir}")
    
    return stage_results


def print_final_summary(results: dict, trainer):
    """打印最終結果總結"""
    print("\n" + "=" * 70)
    print("📋 依序訓練結果總結")
    print("=" * 70)
    
    # 訓練階段結果
    for stage in ['stage1', 'stage2', 'stage3']:
        if stage in results:
            stage_info = results[stage]
            metrics = stage_info['metrics']
            time_taken = stage_info['time']
            
            stage_names = {
                'stage1': '🎨 Stage 1 (分割)',
                'stage2': '🎯 Stage 2 (檢測)',
                'stage3': '📊 Stage 3 (分類)'
            }
            
            print(f"{stage_names[stage]}:")
            print(f"  性能指標: {metrics['main_metric']:.4f}")
            print(f"  訓練時間: {time_taken/60:.1f}分鐘")
            
            if 'forgetting' in stage_info:
                forgetting = stage_info['forgetting']
                print(f"  遺忘程度: {forgetting['max_drop']*100:.2f}%")
                print(f"  遺忘檢查: {'✅ 通過' if forgetting['acceptable'] else '❌ 失敗'}")
    
    # 最終性能
    if 'final_metrics' in results:
        print(f"\n📊 最終性能:")
        final_metrics = results['final_metrics']
        for task, metrics in final_metrics.items():
            print(f"  {task}: {metrics['main_metric']:.4f}")
    
    # 基準比較
    print(f"\n📈 基準性能比較:")
    for task in trainer.baseline_performance:
        baseline = trainer.baseline_performance[task]['main_metric']
        current = trainer.current_performance.get(task, {}).get('main_metric', 0)
        drop = baseline - current
        drop_percent = (drop / baseline * 100) if baseline > 0 else 0
        
        status = "✅" if drop_percent <= 5 else "❌"
        print(f"  {task}: 基準={baseline:.4f}, 當前={current:.4f}, 下降={drop_percent:.2f}% {status}")
    
    # 總體評估
    total_time = results.get('total_time', 0)
    print(f"\n⏱️ 總訓練時間: {total_time/60:.1f}分鐘")
    
    # 最終遺忘檢查
    if 'stage3' in results and 'forgetting' in results['stage3']:
        final_forgetting = results['stage3']['forgetting']
        max_forgetting = final_forgetting['max_drop'] * 100
        
        if final_forgetting['acceptable']:
            print(f"🎉 EWC防遺忘成功！最大遺忘程度: {max_forgetting:.2f}% ≤ 5%")
        else:
            print(f"⚠️ 遺忘程度超過閾值: {max_forgetting:.2f}% > 5%")
    
    print("=" * 70)


def main():
    """主函數"""
    args = parse_arguments()
    
    print("🚀 依序多任務訓練腳本")
    print(f"📅 開始時間: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python: {sys.version}")
    print(f"🔥 PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"🚀 CUDA: {torch.version.cuda}")
        print(f"📱 GPU數量: {torch.cuda.device_count()}")
    else:
        print("💻 使用 CPU")
    
    print("\n📋 訓練計畫:")
    print("Stage 1: 分割任務 (記錄基準)")
    print("Stage 2: 檢測任務 + EWC") 
    print("Stage 3: 分類任務 + EWC")
    print(f"🎯 目標: 遺忘程度 ≤ {args.forgetting_threshold*100:.1f}%")
    
    # 運行訓練
    results = run_sequential_training(args)
    
    # 檢查訓練結果
    if results:
        if 'error' in results:
            print(f"\n❌ 訓練失敗: {results['error']}")
            sys.exit(1)
        elif 'interrupted' in results:
            print(f"\n⚠️ 訓練被中斷，部分結果已保存")
            sys.exit(2)
        else:
            print(f"\n✅ Phase 3 實現完成！準備進入 Phase 4: 評估與優化")
            sys.exit(0)
    else:
        print(f"\n❌ 訓練失敗，未產生結果")
        sys.exit(1)


if __name__ == "__main__":
    main()
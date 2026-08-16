"""阶段1：训练小模型 S（论文：端到端最小化预测误差，只更新 S 与预测头）。

配置按子集自动加载（collm.config.get_config）：
  FD001: 8 层 Transformer Encoder、dropout 0.2；FD003: 6 层、dropout 0.1（同表 I 配置）
  cosine 调度 + lr 2e-3 + batch 256（cosine 13.515→13.007，REVISIONS #45.3）
  多 seed val-best（--seeds；数据划分 seed 固定 42）

用法: python scripts/train_small.py --subset FD001 [--seeds 42 2024 ...]
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from collm.config import get_config
from collm.data import prepare_cmapss
from collm.logging_utils import setup_logger, log_metrics
from collm.models.small_model import SmallModel
from collm.train_common import set_seed, train_loop, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001", choices=["FD001", "FD003"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="训练种子（默认单 seed 42；多 seed 时取 val 最优）")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = get_config(args.subset)
    device = args.device if torch.cuda.is_available() else "cpu"
    tr_ds, val_ds, te_ds, stats = prepare_cmapss(cfg.data, args.subset, cfg.seed)
    log = setup_logger("stage1", args.subset)
    log.info(f"配置: {args.subset} | SM {cfg.small.n_layers} 层 dropout {cfg.small.dropout} "
             f"| 数据 {stats['train_windows']}/{stats['val_windows']}/{stats['test_windows']} 窗口")
    tr_ld = DataLoader(tr_ds, batch_size=cfg.train.batch_size, shuffle=True,
                       num_workers=cfg.train.num_workers, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False)
    te_ld = DataLoader(te_ds, batch_size=cfg.train.batch_size, shuffle=False)

    ckpt_dir = Path(cfg.out_dir) / "checkpoints" / args.subset
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    criterion = nn.MSELoss()
    best_overall = float("inf")
    best_ckpt = None

    for seed in args.seeds:
        set_seed(seed)
        model = SmallModel(cfg.small, max_len=cfg.data.window).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
        scheduler = None
        if cfg.train.sm_sched == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
        ckpt = ckpt_dir / f"small_s{seed}.pt"
        best_val = train_loop(model, tr_ld, val_ld, optimizer, criterion, device,
                              cfg.train.epochs, cfg.train.patience, f"stage1-SM-{seed}",
                              ckpt, logger=log, scheduler=scheduler)
        log_metrics(log, stage="stage1", seed=seed, val_rmse=best_val,
                    params_k=n_params / 1e3)
        if best_val < best_overall:
            best_overall = best_val
            best_ckpt = ckpt

    # 最优 seed 复制为 small.pt（下游 train_conf/evaluate 统一加载）
    shutil.copy(best_ckpt, ckpt_dir / "small.pt")
    model = SmallModel(cfg.small, max_len=cfg.data.window).to(device)
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    te_rmse, te_mae = evaluate(model, te_ld, device)
    log_metrics(log, stage="stage1", selected="best", test_rmse=te_rmse, test_mae=te_mae)
    log.info(f"[SM 最终] test RMSE {te_rmse:.3f} MAE {te_mae:.3f}")


if __name__ == "__main__":
    main()

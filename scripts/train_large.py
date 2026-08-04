"""阶段2：训练大模型 L（论文：冻结 GPT-2 attention+FFN，微调 LN/patch embedding/位置嵌入/预测头）。

配置按子集自动加载（collm.config.get_config）：
  FD001: GPT-2 前 9 层 | FD003: 12 层；patch 4/stride 4；lr 2e-3、batch 256

用法: python scripts/train_large.py --subset FD001
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from collm.config import get_config
from collm.data import prepare_cmapss
from collm.logging_utils import setup_logger, log_metrics
from collm.models.large_model import LargeModel, trainable_params
from collm.train_common import set_seed, train_loop, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001", choices=["FD001", "FD003"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=None, help="训练 seed（默认 cfg.seed=42；数据划分保持 42）")
    args = ap.parse_args()

    cfg = get_config(args.subset)
    set_seed(args.seed if args.seed is not None else cfg.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    tr_ds, val_ds, te_ds, stats = prepare_cmapss(cfg.data, args.subset, cfg.seed)
    log = setup_logger("stage2", args.subset)
    log.info(f"配置: {args.subset} | GPT-2 前 {cfg.large.n_blocks} 层 | "
             f"数据 {stats['train_windows']}/{stats['val_windows']}/{stats['test_windows']} 窗口")
    tr_ld = DataLoader(tr_ds, batch_size=cfg.train.batch_size, shuffle=True,
                       num_workers=cfg.train.num_workers, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False)
    te_ld = DataLoader(te_ds, batch_size=cfg.train.batch_size, shuffle=False)

    model = LargeModel(cfg.large, n_sensors=cfg.data.n_sensors).to(device)
    log.info(f"大模型可训练参数量: {trainable_params(model)/1e3:.1f}K / 总 "
             f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=cfg.train.lr)
    ckpt = Path(cfg.out_dir) / "checkpoints" / args.subset / "large.pt"
    best_val = train_loop(model, tr_ld, val_ld, optimizer, criterion, device,
                          cfg.train.epochs, cfg.train.patience, "stage2-LM", ckpt,
                          logger=log)
    log_metrics(log, stage="stage2", val_rmse=best_val)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    te_rmse, te_mae = evaluate(model, te_ld, device)
    log_metrics(log, stage="stage2", test_rmse=te_rmse, test_mae=te_mae)
    log.info(f"[LM 最终] test RMSE {te_rmse:.3f} MAE {te_mae:.3f}")


if __name__ == "__main__":
    main()

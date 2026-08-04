"""阶段1：训练小模型 S（论文：端到端最小化预测误差，只更新 S 与预测头）。

用法: python scripts/train_small.py --subset FD001 [--epochs 100 --lr 2e-3]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from collm.config import Config
from collm.data import prepare_cmapss
from collm.logging_utils import setup_logger, log_metrics
from collm.models.small_model import SmallModel
from collm.train_common import set_seed, train_loop, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001", choices=["FD001", "FD003"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 2024, 7, 123, 555, 11, 13, 17, 19, 23],
                    help="多个随机种子：各训一次，保留 val 最优（单次训练方差大）")
    ap.add_argument("--pool", default=None, choices=["last", "mean"],
                    help="预测头输入池化方式（实验）")
    ap.add_argument("--layers", type=int, default=None, help="Encoder 层数（实验）")
    ap.add_argument("--head-hidden", type=int, default=None, help="预测头 MLP 隐藏维度（实验）")
    ap.add_argument("--norm-first", action="store_true", help="pre-norm（GPT-2 风格，实验）")
    ap.add_argument("--nheads", type=int, default=None, help="多头数（实验）")
    ap.add_argument("--adamw", action="store_true", help="AdamW + weight decay（实验）")
    ap.add_argument("--wd", type=float, default=1e-4, help="weight decay（--adamw 时生效）")
    ap.add_argument("--dropout", type=float, default=None, help="覆盖 SM dropout（实验）")
    ap.add_argument("--use-val", action="store_true",
                    help="训练集 = train+val 全部窗口（+25% 数据），val 仅用于早停（实验）")
    args = ap.parse_args()

    cfg = Config()
    if args.pool: cfg.small.pool = args.pool
    if args.layers: cfg.small.n_layers = args.layers
    if args.head_hidden: cfg.small.head_hidden = args.head_hidden
    if args.norm_first: cfg.small.norm_first = True
    if args.nheads: cfg.small.n_heads = args.nheads
    device = args.device if torch.cuda.is_available() else "cpu"

    tr_ds, val_ds, te_ds, stats = prepare_cmapss(cfg.data, args.subset, args.seeds[0])
    if args.dropout is not None:
        cfg.small.dropout = args.dropout
    if args.use_val:
        from torch.utils.data import ConcatDataset
        tr_ds = ConcatDataset([tr_ds, val_ds])     # +25% 训练数据，val 仅早停
    log = setup_logger("stage1", args.subset)
    log.info(f"数据统计: {stats}")
    tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                       num_workers=cfg.train.num_workers, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False)
    te_ld = DataLoader(te_ds, batch_size=args.batch, shuffle=False)

    ckpt_dir = Path(cfg.out_dir) / "checkpoints" / args.subset
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    criterion = nn.MSELoss()
    best_overall = float("inf")
    best_ckpt = ckpt_dir / "small.pt"

    for seed in args.seeds:
        set_seed(seed)
        model = SmallModel(cfg.small, max_len=cfg.data.window).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        if args.adamw:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        ckpt = ckpt_dir / f"small_s{seed}.pt"
        best_val = train_loop(model, tr_ld, val_ld, optimizer, criterion, device,
                              args.epochs, cfg.train.patience, f"stage1-SM-{seed}", ckpt,
                              logger=log)
        log_metrics(log, stage="stage1", seed=seed, val_rmse=best_val,
                    params_k=n_params / 1e3)
        if best_val < best_overall:
            best_overall = best_val
            best_ckpt = ckpt

    log.info(f"最优 seed 模型: {best_ckpt}（val RMSE {best_overall:.4f}）")
    # 统一接口：最优 seed 复制为 small.pt（下游 train_conf/evaluate 加载）
    import shutil
    shutil.copy(best_ckpt, ckpt_dir / "small.pt")
    model = SmallModel(cfg.small, max_len=cfg.data.window).to(device)
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    te_rmse, te_mae = evaluate(model, te_ld, device)
    log_metrics(log, stage="stage1", selected="best", test_rmse=te_rmse, test_mae=te_mae)
    log.info(f"[SM 最终] test RMSE {te_rmse:.3f} MAE {te_mae:.3f}")


if __name__ == "__main__":
    main()

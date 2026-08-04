"""阶段2：训练大模型 L（论文：冻结 GPT-2 自注意力+FFN，训练 patch embedding 与预测头）。

用法: python scripts/train_large.py --subset FD001 [--lr 2e-3 --epochs 100]
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
from collm.models.large_model import LargeModel
from collm.train_common import set_seed, train_loop, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001", choices=["FD001", "FD003"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3, help="LM 微调 lr（表 I 的 2e-3 对冻结骨干偏大，见 REVISIONS #5）")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--freeze-ln-f", action="store_true",
                    help="冻结 ln_f（含逐层 LN 全部冻结——FD003 实测指标更好）")
    ap.add_argument("--instance-norm", action="store_true",
                    help="patch 前 per-sample per-channel 归一化（One Fits All 核心组件，实验）")
    ap.add_argument("--pool", default="last", choices=["last", "mean"],
                    help="预测头输入：末端 patch 或全部 patch 均值（实验）")
    ap.add_argument("--blocks", type=int, default=None,
                    help="使用的 GPT-2 层数（One Fits All 官方 6 层，实验）")
    ap.add_argument("--linear-head", action="store_true",
                    help="预测头用单层 Linear（One Fits All 原版；默认 MLP）")
    ap.add_argument("--no-dropout", action="store_true",
                    help="关闭 GPT-2 注意力/残差 dropout（实验）")
    ap.add_argument("--pad-patches", action="store_true",
                    help="尾部复制补足完整 patch（12.5→13 个，实验）")
    ap.add_argument("--two-stage", action="store_true",
                    help="两阶段（One Fits All linear-probe→fine-tune）：先只训 patch_embed 1 epoch，再全量微调")
    ap.add_argument("--use-val", action="store_true",
                    help="训练集 = train+val 全部窗口（+25% 数据），val 仅用于早停（实验）")
    ap.add_argument("--head-concat", action="store_true",
                    help="GPT4TS 官方头：全部 patch 特征拼接 → Linear(768×n → 1)（实验）")
    ap.add_argument("--ci", action="store_true",
                    help="GPT4TS 官方：通道独立（每通道独立过 GPT-2，14 通道预测平均）（实验）")
    ap.add_argument("--pad-zero", action="store_true",
                    help="尾部零填充补 patch（BANjian16 复现；默认 replicate 复制边缘）")
    ap.add_argument("--adamw", action="store_true", help="AdamW + weight decay（BANjian16 配置）")
    ap.add_argument("--wd", type=float, default=1e-4, help="weight decay（--adamw 时生效）")
    ap.add_argument("--cosine", action="store_true", help="CosineAnnealingLR（T_max=epochs）")
    ap.add_argument("--input-norm", action="store_true",
                    help="proj 后加 LayerNorm（BANjian16 的 input_norm，实验）")
    ap.add_argument("--ckpt-suffix", default="", help="checkpoint 文件名后缀（防实验互相覆盖）")
    args = ap.parse_args()

    cfg = Config()
    if args.freeze_ln_f:
        cfg.large.learnable_ln_f = False
        # 逐层 ln_1/ln_2 一并冻结（LargeModel 内按 learnable_ln_f 控制）
    if args.instance_norm:
        cfg.large.instance_norm = True
    cfg.large.pool_mode = args.pool
    if args.blocks is not None:
        cfg.large.n_blocks = args.blocks
    if args.linear_head:
        cfg.large.head_hidden = None
    if args.no_dropout:
        cfg.large.dropout = 0.0
    if args.pad_patches:
        cfg.large.pad_patches = True
    if args.head_concat:
        cfg.large.head_concat = True
    if args.ci:
        cfg.large.channel_independent = True
    if args.pad_zero:
        cfg.large.pad_mode = "zero"
    if args.input_norm:
        cfg.large.input_norm = True
    set_seed(cfg.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    tr_ds, val_ds, te_ds, stats = prepare_cmapss(cfg.data, args.subset, cfg.seed)
    if args.use_val:
        from torch.utils.data import ConcatDataset
        tr_ds = ConcatDataset([tr_ds, val_ds])     # +25% 训练数据，val 仅早停
    tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                       num_workers=cfg.train.num_workers, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False)
    te_ld = DataLoader(te_ds, batch_size=args.batch, shuffle=False)

    model = LargeModel(cfg.large, n_sensors=cfg.data.n_sensors).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log = setup_logger("stage2", args.subset)
    log.info(f"大模型可训练参数量: {n_train/1e3:.1f}K / 总 "
             f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    criterion = nn.MSELoss()
    if args.two_stage:
        # 阶段A（linear probing，1 epoch）：只训 patch_embed，其余全冻结
        for p in model.parameters():
            p.requires_grad = False
        for p in model.patch_embed.parameters():
            p.requires_grad = True
        opt_p = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        log.info("two-stage A：只训 patch_embed（linear probing，1 epoch）")
        probe_ckpt = Path(cfg.out_dir) / "checkpoints" / args.subset / "large_probe.pt"
        train_loop(model, tr_ld, val_ld, opt_p, criterion, device, 1,
                   cfg.train.patience, "stage2-probe", probe_ckpt, logger=log)
        # 阶段B：放开 LN/pos/head 全量微调
        for p in model.parameters():
            p.requires_grad = False
        for p in model.gpt2.parameters():
            if cfg.large.learnable_ln_f:
                for blk in model.gpt2.h:
                    for q in list(blk.ln_1.parameters()) + list(blk.ln_2.parameters()):
                        q.requires_grad = True
                for q in model.gpt2.ln_f.parameters():
                    q.requires_grad = True
        for p in model.patch_embed.parameters():
            p.requires_grad = True
        model.pos_embed.requires_grad = True
        for p in model.predictor.parameters():
            p.requires_grad = True
        log.info("two-stage B：全量微调（patch_embed+pos+LN+head）")
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    else:
        if args.adamw:
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.wd)
        else:
            optimizer = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    if args.cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    else:
        scheduler = None
    ckpt = Path(cfg.out_dir) / "checkpoints" / args.subset / f"large{args.ckpt_suffix}.pt"
    best_val = train_loop(model, tr_ld, val_ld, optimizer, criterion, device,
                          args.epochs, cfg.train.patience, "stage2-LM", ckpt,
                          logger=log, scheduler=scheduler)
    log_metrics(log, stage="stage2", val_rmse=best_val)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    te_rmse, te_mae = evaluate(model, te_ld, device)
    log_metrics(log, stage="stage2", test_rmse=te_rmse, test_mae=te_mae)
    log.info(f"[LM 最终] test RMSE {te_rmse:.3f} MAE {te_mae:.3f}")


if __name__ == "__main__":
    main()

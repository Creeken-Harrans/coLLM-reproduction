"""SM 训练配方消融（实验区）。

定稿 SM 只用 plain Adam lr 2e-3 + 早停；余弦调度与 weight decay 从未测试。
FD003 的 5-seed val-best 曾带来 0.6 的改善（11.78→11.16），FD001 只跑过
单 seed——多 seed val-best 是防运气策略（REVISIONS #10）。

变体：--layers/--dropout/--lr/--sched/--wd/--seed（数据划分 seed 固定 42）
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from collm.config import get_config
from collm.data import prepare_cmapss
from collm.models.small_model import SmallModel
from collm.training import set_seed, rmse_mae


def run(args):
    t0 = time.time()
    cfg = get_config(args.subset)
    cfg.small.n_layers = args.layers
    cfg.small.dropout = args.dropout
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr_ds, val_ds, te_ds, stats = prepare_cmapss(cfg.data, args.subset, cfg.seed)
    tr_ld = DataLoader(tr_ds, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=256, shuffle=False)
    te_ld = DataLoader(te_ds, batch_size=256, shuffle=False)

    model = SmallModel(cfg.small, max_len=cfg.data.window).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = None
    if args.sched == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        sched = CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.MSELoss()

    best_val, best_ep, bad = float("inf"), -1, 0
    ckpt = f"/tmp/sm_sweep_{__import__('os').getpid()}.pt"
    for ep in range(1, args.epochs + 1):
        model.train()
        for x, y in tr_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x)[0], y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        if sched is not None:
            sched.step()
        model.eval()
        with torch.no_grad():
            vp, vt = [], []
            for x, y in val_ld:
                vp.append(model(x.to(device))[0].cpu()); vt.append(y)
            val_rmse = rmse_mae(torch.cat(vp), torch.cat(vt))[0]
        improved = val_rmse < best_val
        if improved:
            best_val, best_ep = val_rmse, ep
            torch.save(model.state_dict(), ckpt)
            bad = 0
        else:
            bad += 1
        if ep % 10 == 0 or improved:
            print(f"  ep {ep:3d} | val {val_rmse:.3f}" + (" *" if improved else ""))
        if bad >= args.patience:
            print(f"  早停 ep {ep}（best ep {best_ep} val {best_val:.3f}）")
            break

    model.load_state_dict(torch.load(ckpt))
    model.eval()
    with torch.no_grad():
        tp, tt = [], []
        for x, y in te_ld:
            tp.append(model(x.to(device))[0].cpu()); tt.append(y)
        test_rmse, test_mae = rmse_mae(torch.cat(tp), torch.cat(tt))

    tag = (f"{args.subset}_L{args.layers}_d{args.dropout}_lr{args.lr}_wd{args.wd}"
           f"_s{args.sched}_seed{args.seed}")
    out = {"tag": tag, "best_val": best_val, "best_epoch": best_ep,
           "test_rmse": test_rmse, "test_mae": test_mae, "seconds": time.time() - t0}
    path = Path("experiments/results") / f"{tag}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"  → {tag}: best-val {best_val:.3f}@ep{best_ep} | TEST {test_rmse:.3f}/{test_mae:.3f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001", choices=["FD001", "FD003"])
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--sched", default="none", choices=["none", "cosine"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args)

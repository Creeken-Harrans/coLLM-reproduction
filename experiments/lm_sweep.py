"""LM 架构/训练配方消融（实验区，独立于 collm/ 定稿代码）。

目标：定位论文 One Fits All Fine-tuning（FD001 12.34 / FD003 11.18）的
可复现配置。变体按 CLI 开关控制，一次一跑，结果落 experiments/results/。

变体维度：
  --layers N           GPT-2 块数（6/9/12）
  --patch N            patch 大小（默认 4）
  --pad {none,replicate,zero}   patch 尾部补全方式
  --head {last,mean,concat,mlp} 预测头
  --finetune {ln,all}  冻结范围：ln=仅 LN/wpe/patch/头（论文 III-D），
                       all=全量微调（表 II Fine-tuning 解读）
  --instancenorm       输入逐样本 instance norm（无 denorm，RUL 目标不符 denorm）
  --lr LR              学习率
  --wd WD              Adam weight decay
  --batch N            批大小
  --epochs N           最大 epoch（早停 patience --patience）
  --sched {none,cosine,type1}  调度
  --seed S             训练 seed（数据划分 seed 固定 42）
  --subset FD001|FD003

用法示例:
  python experiments/lm_sweep.py --subset FD001 --finetune all --lr 1e-4 --wd 0.01 --sched cosine
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Model, GPT2Config

from collm.config import get_config
from collm.data import prepare_cmapss
from collm.training import set_seed, rmse_mae


def build_model(cfg, args):
    """按变体开关构造 LM。"""
    gpt2 = GPT2Model.from_pretrained(cfg.large.model_name)
    if args.layers < len(gpt2.h):
        gpt2.h = gpt2.h[:args.layers]

    if args.finetune == "ln":
        for p in gpt2.parameters():
            p.requires_grad = False
        for block in gpt2.h:
            for p in block.ln_1.parameters():
                p.requires_grad = True
            for p in block.ln_2.parameters():
                p.requires_grad = True
        for p in gpt2.ln_f.parameters():
            p.requires_grad = True
    # 'all'：全部可训练（表 II Fine-tuning 解读）

    in_dim = args.patch * cfg.data.n_sensors
    patch_embed = nn.Linear(in_dim, cfg.large.d_embed)
    pos_embed = nn.Parameter(gpt2.wpe.weight[:64].unsqueeze(0).clone().detach())
    gpt2.wpe.requires_grad_(False)

    n_patch_eff = 13 if args.pad == "replicate" else 12  # 窗口 50 / patch 4
    if args.head == "concat":
        # GPT4TS 官方标量头：GELU → 展平全部 patch → LayerNorm → Linear
        head = nn.Sequential(
            nn.GELU(),
            nn.LayerNorm(cfg.large.d_embed * n_patch_eff),
            nn.Linear(cfg.large.d_embed * n_patch_eff, 1))
    elif args.head == "concat_plain":
        head = nn.Linear(cfg.large.d_embed * n_patch_eff, 1)
    elif args.head == "mlp":
        head = nn.Sequential(nn.Linear(cfg.large.d_embed, 128), nn.GELU(),
                             nn.Dropout(0.1), nn.Linear(128, 1))
    else:
        head = nn.Linear(cfg.large.d_embed, 1)

    class LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.gpt2 = gpt2
            self.patch_embed = patch_embed
            self.pos_embed = pos_embed
            self.head = head
            self.args = args
            self.n_sensors = cfg.data.n_sensors

        def _patchify(self, x):
            ps = args.patch
            pat = x.unfold(1, ps, ps)
            pat = pat.permute(0, 1, 3, 2).reshape(x.shape[0], -1, ps * self.n_sensors)
            if args.pad == "replicate":
                last = pat[:, -1:]
                pat = torch.cat([pat, last], dim=1)   # GPT4TS 官方尾补 → 13 patch
            elif args.pad == "zero":
                pat = torch.cat([pat, torch.zeros_like(pat[:, :1])], dim=1)
            return pat

        def forward(self, x, return_feat=True):
            x = self._patchify(x)
            x = self.patch_embed(x)
            x = x + self.pos_embed[:, : x.shape[1]]
            h = x
            for block in self.gpt2.h:
                h = block(h)
            h = self.gpt2.ln_f(h)
            if args.head in ("concat", "concat_plain"):
                z = h.reshape(h.shape[0], -1)
            elif args.head == "mean":
                z = h.mean(dim=1)
            else:
                z = h[:, -1]
            y = self.head(z).squeeze(-1)
            if return_feat:
                return y, h
            return y

    m = LM()
    if args.instancenorm:
        class INWrap(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
            def forward(self, x, return_feat=True):
                mu = x.mean(dim=1, keepdim=True)
                std = x.std(dim=1, keepdim=True) + 1e-5
                return self.inner((x - mu) / std, return_feat=return_feat)
        m = INWrap(m)
    return m


def run(args):
    t0 = time.time()
    cfg = get_config(args.subset)
    set_seed(args.seed if args.seed is not None else cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr_ds, val_ds, te_ds, stats = prepare_cmapss(cfg.data, args.subset, cfg.seed)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=256, shuffle=False)
    te_ld = DataLoader(te_ds, batch_size=256, shuffle=False)

    model = build_model(cfg, args).to(device)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[{args.subset}] 变体: {args} | 可训练 {n_tr/1e3:.0f}K / 总 {n_all/1e6:.1f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.sched == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        sched = CosineAnnealingLR(opt, T_max=args.epochs)
    elif args.sched == "type1":
        from torch.optim.lr_scheduler import LambdaLR
        sched = LambdaLR(opt, lambda e: 1.0 if e < 3 else 0.9 ** (e - 3))
    else:
        sched = None
    crit = nn.MSELoss()

    best_val, best_ep, bad = float("inf"), -1, 0
    ckpt_tmp = f"/tmp/lm_sweep_{os.getpid()}.pt"
    hist = []
    for ep in range(1, args.epochs + 1):
        model.train()
        for x, y in tr_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x, return_feat=False), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        if sched is not None:
            sched.step()
        model.eval()
        with torch.no_grad():
            vp, vt, tp, tt = [], [], [], []
            for x, y in val_ld:
                vp.append(model(x.to(device), return_feat=False).cpu())
                vt.append(y)
            val_rmse = rmse_mae(torch.cat(vp), torch.cat(vt))[0]
            for x, y in te_ld:
                tp.append(model(x.to(device), return_feat=False).cpu())
                tt.append(y)
            test_rmse = rmse_mae(torch.cat(tp), torch.cat(tt))[0]
        hist.append({"epoch": ep, "val": float(val_rmse), "test": float(test_rmse)})
        improved = val_rmse < best_val
        if improved:
            best_val, best_ep = val_rmse, ep
            torch.save(model.state_dict(), ckpt_tmp)
            bad = 0
        else:
            bad += 1
        if ep % 10 == 0 or improved:
            print(f"  ep {ep:3d} | val {val_rmse:.3f} test {test_rmse:.3f}" + (" *" if improved else ""))
        if bad >= args.patience:
            print(f"  早停 ep {ep}（best ep {best_ep} val {best_val:.3f}）")
            break

    # 最终：加载 best-val 权重评估 test
    model.load_state_dict(torch.load(ckpt_tmp))
    model.eval()
    with torch.no_grad():
        tp, tt = [], []
        for x, y in te_ld:
            tp.append(model(x.to(device), return_feat=False).cpu())
            tt.append(y)
        test_rmse, test_mae = rmse_mae(torch.cat(tp), torch.cat(tt))

    tag = (f"{args.subset}_L{args.layers}_p{args.patch}_pad{args.pad}_head{args.head}"
           f"_ft{args.finetune}_in{int(args.instancenorm)}_lr{args.lr}_wd{args.wd}"
           f"_b{args.batch}_ep{args.epochs}_s{args.sched}_seed{args.seed}")
    out = {"tag": tag, "best_val": best_val, "best_epoch": best_ep,
           "test_rmse": test_rmse, "test_mae": test_mae, "trainable_k": n_tr / 1e3,
           "seconds": time.time() - t0, "history": hist}
    path = Path("experiments/results") / f"{tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"  → {tag}: best-val {best_val:.3f} @ep{best_ep} | TEST RMSE {test_rmse:.3f} MAE {test_mae:.3f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001", choices=["FD001", "FD003"])
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--pad", default="none", choices=["none", "replicate", "zero"])
    ap.add_argument("--head", default="mlp", choices=["last", "mean", "concat", "concat_plain", "mlp"])
    ap.add_argument("--finetune", default="ln", choices=["ln", "all"])
    ap.add_argument("--instancenorm", action="store_true")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--sched", default="none", choices=["none", "cosine", "type1"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args)

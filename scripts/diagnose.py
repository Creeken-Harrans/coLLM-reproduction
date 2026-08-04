"""置信度校准诊断：Qs/Ql 分布、与真实误差相关性、隶属函数学习情况。

用法: python scripts/diagnose.py --subset FD001
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from collm.config import get_config
from collm.data import prepare_cmapss
from collm.models.collm import CoLLM


@torch.no_grad()
def collect(model: CoLLM, loader, device):
    ys, yl, qs, ql, yt = [], [], [], [], []
    model.eval()
    for x, y in loader:
        x = x.to(device)
        ys_b, fs = model.small(x)
        yl_b, fl = model.large(x)
        qs_b, ql_b = model.fuzzy(fs), model.reflection(fl)
        ys.append(ys_b.cpu()); yl.append(yl_b.cpu())
        qs.append(qs_b.cpu()); ql.append(ql_b.cpu()); yt.append(y)
    return [torch.cat(v).numpy() for v in (ys, yl, qs, ql, yt)]


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra, rb = ra.astype(float), rb.astype(float)
    n = len(ra)
    ma, mb = ra.mean(), rb.mean()
    num = ((ra - ma) * (rb - mb)).sum()
    den = np.sqrt(((ra - ma) ** 2).sum() * ((rb - mb) ** 2).sum())
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001", choices=["FD001", "FD003"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = get_config(args.subset)
    device = args.device if torch.cuda.is_available() else "cpu"
    alpha = cfg.fuzzy.alpha
    ckpt = Path(cfg.out_dir) / "checkpoints" / args.subset

    # 从权重推断层数 + 阶段3 配置对齐（子代理审查发现：此前固定默认值导致加载崩溃）
    ckpt_dir = Path(cfg.out_dir) / "checkpoints" / args.subset
    _sd = torch.load(ckpt_dir / "small.pt", map_location="cpu")
    cfg.small.n_layers = max(int(k.split(".")[2]) for k in _sd if "encoder.layers." in k) + 1
    _sd2 = torch.load(ckpt_dir / "large.pt", map_location="cpu")
    cfg.large.n_blocks = max(int(k.split(".")[2]) for k in _sd2 if "gpt2.h." in k) + 1

    model = CoLLM(cfg).to(device).eval()
    for part, name in [("small", "small"), ("large", "large"),
                       ("fuzzy", "fuzzy"), ("reflection", "reflection")]:
        p = ckpt / f"{name}.pt"
        if p.exists():
            getattr(model, part).load_state_dict(torch.load(p, map_location=device))
        else:
            print(f"警告: 缺少 {p}")

    _, _, te_ds, _ = prepare_cmapss(cfg.data, args.subset, cfg.seed)
    te_ld = DataLoader(te_ds, batch_size=512, shuffle=False)
    ys, yl, qs, ql, yt = collect(model, te_ld, device)

    err_s = np.abs(ys - yt); err_l = np.abs(yl - yt)
    qs_star = 1 - np.tanh(err_s / alpha); ql_star = 1 - np.tanh(err_l / alpha)

    print(f"=== {args.subset} 置信度校准诊断（α={alpha}）===")
    for name, q, q_star, err in [("Qs(SM)", qs, qs_star, err_s), ("Ql(LM)", ql, ql_star, err_l)]:
        print(f"\n[{name}] 预测分布: min {q.min():.3f} | p25 {np.percentile(q,25):.3f} | "
              f"中位 {np.median(q):.3f} | p75 {np.percentile(q,75):.3f} | max {q.max():.3f}")
        print(f"  标签分布: p25 {np.percentile(q_star,25):.3f} | 中位 {np.median(q_star):.3f} | "
              f"p75 {np.percentile(q_star,75):.3f}")
        print(f"  预测-标签 MSE: {np.mean((q-q_star)**2):.4f} | "
              f"Spearman(q, 1/误差): {spearman(q, -err):.3f} | Spearman(q, 误差): {spearman(q, err):.3f}")
        # 分箱 RMSE（图 5 风格）
        order = np.argsort(q)
        edges = np.linspace(0, len(q), 11).astype(int)
        bins = []
        for i in range(10):
            idx = order[edges[i]:edges[i+1]]
            bins.append(np.sqrt(np.mean(err[idx]**2)))
        print(f"  十等分箱 RMSE: {' '.join(f'{b:.1f}' for b in bins)}")

    # 隶属函数学习情况
    mu = model.fuzzy.membership.mu.detach().cpu().numpy()
    sigma = np.exp(model.fuzzy.membership.log_sigma.detach().cpu().numpy())
    print(f"\n[隶属函数] μ 范围 [{mu.min():.2f}, {mu.max():.2f}] | σ 范围 "
          f"[{sigma.min():.3f}, {sigma.max():.3f}] 中位 {np.median(sigma):.3f}")
    # 模糊特征饱和率（隶属度 ≈0 或 ≈1 的比例）
    with torch.no_grad():
        x, _ = te_ds[0]
        _, fs = model.small(x.unsqueeze(0).to(device))
        m = model.fuzzy.membership(fs)
        sat = ((m < 0.05) | (m > 0.95)).float().mean().item()
    print(f"  模糊隶属度饱和率（≈0 或 ≈1）: {sat*100:.1f}%")


if __name__ == "__main__":
    main()

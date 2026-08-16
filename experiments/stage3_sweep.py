"""阶段3 α/聚合方式扫描（实验区）。

前提：outputs/checkpoints/{subset}/small.pt 与 large.pt 已存在（阶段1/2 定稿）。

理论（2026-08-15 推导）：
  论文 Q* = 1 - tanh(|e|/α)。α 决定路由行为。由表 II FLOPs 加速比反推
  论文各阈值下大模型触发率（LM=2.21G，SM+FNN≈0.01G）：
    FD001: A 25.3% / B 47.6% / C 78.9%    FD003: A 6.4% / B 43.2% / C 63.2%
  校准假设下 P(Q*<τ)=P(|e|>α·atanh(1-τ))（半正态 |e|，σ≈MAE·√(π/2)）：
    α=10 时 FD003 16/44/64%、FD001 43/53/70% —— 与论文锚点最接近
  本扫描在 α∈{6,8,10,12,15} × pool∈{mean,stats} 上训练 FNN/R，
  以 val 组合 RMSE 为选择依据（测试锚点仅作最终报告，不参与选择——防泄漏）。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, TensorDataset

from collm.config import get_config
from collm.data import prepare_cmapss
from collm.models.collm import CoLLM
from collm.models.fuzzy import FuzzyAgent, confidence_label
from collm.models.reflection import ReflectionModel
from collm.train_common import set_seed


def combine(ys, yl, qs, ql, tau1, tau2, use_reflection=True):
    ys, yl, qs, ql = map(torch.as_tensor, (ys, yl, qs, ql))
    yf = ys.clone()
    need = qs < tau1
    reflect = torch.zeros_like(need)
    if use_reflection and need.any():
        delta = qs[need] - ql[need]
        refl = delta > tau2
        reflect[need] = refl
        yf[need] = torch.where(refl, (ys[need] + yl[need]) / 2.0, yl[need])
    elif need.any():
        yf[need] = yl[need]
    return yf, need, reflect


def rmse(a, b):
    return float(torch.sqrt(((a - b) ** 2).mean()))


def mae(a, b):
    return float((a - b).abs().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001", choices=["FD001", "FD003"])
    ap.add_argument("--alphas", type=float, nargs="+", default=[6, 8, 10, 12, 15])
    ap.add_argument("--pools", nargs="+", default=["mean", "stats"])
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()

    cfg = get_config(args.subset)
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = Path(cfg.out_dir) / "checkpoints" / args.subset

    model = CoLLM(cfg).to(device)
    model.small.load_state_dict(torch.load(ckpt_dir / "small.pt", map_location=device))
    model.large.load_state_dict(torch.load(ckpt_dir / "large.pt", map_location=device))
    for m in (model.small, model.large):
        for p in m.parameters():
            p.requires_grad = False
    model.eval()

    tr_ds, val_ds, te_ds, stats = prepare_cmapss(cfg.data, args.subset, cfg.seed)

    @torch.no_grad()
    def extract(ds):
        ld = DataLoader(ds, batch_size=cfg.train.conf_batch, shuffle=False, num_workers=0)
        fs, fl, ys, yl, yt = [], [], [], [], []
        for x, y in ld:
            x = x.to(device)
            ys_b, fs_b = model.small(x)
            yl_b, fl_b = model.large(x)
            fs.append(fs_b.cpu()); fl.append(fl_b.cpu())
            ys.append(ys_b.cpu()); yl.append(yl_b.cpu()); yt.append(y)
        return (torch.cat(fs), torch.cat(fl), torch.cat(ys), torch.cat(yl), torch.cat(yt))

    ftr = extract(tr_ds)
    fva = extract(val_ds)
    fte = extract(te_ds)

    rows = []
    for pool in args.pools:
        for alpha in args.alphas:
            cfg.fuzzy.pool_mode = pool
            cfg.fuzzy.alpha = alpha
            cfg.reflection.alpha = alpha
            set_seed(cfg.seed)   # 每组合独立 seed：与 train_conf 完全一致
            init_mu, init_sigma = FuzzyAgent.init_from_features(
                ftr[0], cfg.fuzzy.n_membership // cfg.fuzzy.d_input)
            fuzzy = FuzzyAgent(cfg.fuzzy, init_mu=init_mu, init_sigma=init_sigma).to(device)
            refl = ReflectionModel(cfg.reflection).to(device)
            opt = torch.optim.Adam(list(fuzzy.parameters()) + list(refl.parameters()),
                                   lr=cfg.train.conf_lr)
            crit = torch.nn.MSELoss()
            # 训练数据 = train+val；DataLoader 构造与 train_conf 逐位一致
            # （batch/每 epoch 重洗/num_workers=0——Ql 饱和悬崖对配方敏感，REVISIONS #48）
            qs_t = confidence_label(ftr[2], ftr[4], alpha)
            ql_t = confidence_label(ftr[3], ftr[4], alpha)
            qs_v = confidence_label(fva[2], fva[4], alpha)
            ql_v = confidence_label(fva[3], fva[4], alpha)
            F = torch.cat([ftr[0], fva[0]])
            L = torch.cat([ftr[1], fva[1]])
            QS = torch.cat([qs_t, qs_v]); QL = torch.cat([ql_t, ql_v])
            loader = DataLoader(TensorDataset(F, L, QS, QL),
                                batch_size=cfg.train.conf_batch, shuffle=True,
                                num_workers=0)
            fuzzy.train(); refl.train()
            for ep in range(args.epochs):
                for f_, l_, qs_, ql_ in loader:
                    f_, l_, qs_, ql_ = (f_.to(device), l_.to(device),
                                        qs_.to(device), ql_.to(device))
                    opt.zero_grad(set_to_none=True)
                    loss = crit(fuzzy(f_), qs_) + crit(refl(l_), ql_)
                    loss.backward(); opt.step()
            fuzzy.eval(); refl.eval()
            with torch.no_grad():
                qs_va = fuzzy(fva[0].to(device)).cpu()
                ql_va = refl(fva[1].to(device)).cpu()
                qs_te = fuzzy(fte[0].to(device)).cpu()
                ql_te = refl(fte[1].to(device)).cpu()
            ths = cfg.thresholds
            combos = {
                "A": (ths.fd001_a if args.subset == "FD001" else ths.fd003_a),
                "B": (ths.fd001_b if args.subset == "FD001" else ths.fd003_b),
                "C": (ths.fd001_c if args.subset == "FD001" else ths.fd003_c),
            }
            row = {"alpha": alpha, "pool": pool, "val": {}, "test": {}}
            for name, (t1, t2) in combos.items():
                yf, need, refl_m = combine(fva[2], fva[3], qs_va, ql_va, t1, t2)
                row["val"][name] = {"rmse": rmse(yf, fva[4]),
                                    "lm_ratio": float(need.float().mean()),
                                    "refl_ratio": float(refl_m.float().mean())}
                yf, need, refl_m = combine(fte[2], fte[3], qs_te, ql_te, t1, t2)
                row["test"][name] = {"rmse": rmse(yf, fte[4]), "mae": mae(yf, fte[4]),
                                     "lm_ratio": float(need.float().mean()),
                                     "refl_ratio": float(refl_m.float().mean())}
            score = sum(row["val"][n]["rmse"] for n in combos)
            row["val_score"] = score
            rows.append(row)
            print(f"α={alpha} pool={pool}: val A/B/C "
                  f"{row['val']['A']['rmse']:.3f}/{row['val']['B']['rmse']:.3f}/{row['val']['C']['rmse']:.3f} "
                  f"| test A {row['test']['A']['rmse']:.3f} B {row['test']['B']['rmse']:.3f} C {row['test']['C']['rmse']:.3f} "
                  f"| test lm_ratio {row['test']['A']['lm_ratio']*100:.0f}/{row['test']['B']['lm_ratio']*100:.0f}/{row['test']['C']['lm_ratio']*100:.0f}% "
                  f"| test refl {row['test']['C']['refl_ratio']*100:.1f}%")

    out = Path("experiments/results") / f"stage3_{args.subset}.json"
    out.write_text(json.dumps(rows, indent=1))
    rows.sort(key=lambda r: r["val_score"])
    print("\n按 val_score 排序最优:", rows[0]["alpha"], rows[0]["pool"])
    print(f"结果: {out}")


if __name__ == "__main__":
    main()

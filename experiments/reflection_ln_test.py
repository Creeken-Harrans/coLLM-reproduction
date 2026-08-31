"""反思网络 ±LayerNorm 对照（α 定稿后的一次性校准测试）。

论文公式 12：展平 + 单层全连接。REVISIONS #14 曾在 α=4/5 下测试 LN 后移除
（#42：去 LN 组合改善 0.4）。α=6/10（测试时定稿）后目标分布改变（Q*_l 均值 0.34/0.71），
重测 LN 是否恢复 test 端 Ql 校准（当前 Ql 在 test 饱和至 0 → 反思无选择性）。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from collm.config import get_config
from collm.data import prepare_cmapss
from collm.models.collm import CoLLM
from collm.models.fuzzy import FuzzyAgent, confidence_label
from collm.models.reflection import ReflectionModel
from collm.training import set_seed, rmse_mae


def combine(ys, yl, qs, ql, tau1, tau2):
    ys, yl, qs, ql = map(torch.as_tensor, (ys, yl, qs, ql))
    yf = ys.clone()
    need = qs < tau1
    refl = torch.zeros_like(need)
    if need.any():
        delta = qs[need] - ql[need]
        r = delta > tau2
        refl[need] = r
        yf[need] = torch.where(r, (ys[need] + yl[need]) / 2.0, yl[need])
    return yf, need, refl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001", choices=["FD001", "FD003"])
    ap.add_argument("--ln", action="store_true")
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

    tr_ds, val_ds, te_ds, _ = prepare_cmapss(cfg.data, args.subset, cfg.seed)

    @torch.no_grad()
    def extract(ds, batch=1024):
        ld = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
        fs, fl, ys, yl, yt = [], [], [], [], []
        for x, y in ld:
            x = x.to(device)
            ys_b, fs_b = model.small(x)
            yl_b, fl_b = model.large(x)
            fs.append(fs_b.cpu()); fl.append(fl_b.cpu())
            ys.append(ys_b.cpu()); yl.append(yl_b.cpu()); yt.append(y)
        return (torch.cat(fs), torch.cat(fl), torch.cat(ys), torch.cat(yl), torch.cat(yt))

    ftr = extract(tr_ds); fva = extract(val_ds); fte = extract(te_ds)
    alpha = cfg.fuzzy.alpha

    init_mu, init_sigma = FuzzyAgent.init_from_features(
        ftr[0], cfg.fuzzy.n_membership // cfg.fuzzy.d_input)
    fuzzy = FuzzyAgent(cfg.fuzzy, init_mu=init_mu, init_sigma=init_sigma).to(device)
    refl = ReflectionModel(cfg.reflection).to(device)
    if args.ln:
        class ReflLN(nn.Module):
            def __init__(self, inner, d):
                super().__init__()
                self.ln = nn.LayerNorm(d)
                self.inner = inner
            def forward(self, feat):
                return self.inner(self.ln(feat.reshape(feat.shape[0], -1)).reshape(feat.shape))
        refl = ReflLN(refl, cfg.reflection.d_input).to(device)
    opt = torch.optim.Adam(list(fuzzy.parameters()) + list(refl.parameters()), lr=cfg.train.conf_lr)
    crit = nn.MSELoss()

    qs_t = confidence_label(ftr[2], ftr[4], alpha); ql_t = confidence_label(ftr[3], ftr[4], alpha)
    qs_v = confidence_label(fva[2], fva[4], alpha); ql_v = confidence_label(fva[3], fva[4], alpha)
    F = torch.cat([ftr[0], fva[0]]); L = torch.cat([ftr[1], fva[1]])
    QS = torch.cat([qs_t, qs_v]); QL = torch.cat([ql_t, ql_v])
    perm = torch.randperm(len(F))
    fuzzy.train(); refl.train()
    for ep in range(args.epochs):
        for i in range(0, len(F), 2048):
            idx = perm[i:i + 2048]
            f_, l_, qs_, ql_ = F[idx].to(device), L[idx].to(device), QS[idx].to(device), QL[idx].to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(fuzzy(f_), qs_) + crit(refl(l_), ql_)
            loss.backward(); opt.step()
    fuzzy.eval(); refl.eval()
    with torch.no_grad():
        qs_te = fuzzy(fte[0].to(device)).cpu(); ql_te = refl(fte[1].to(device)).cpu()
        qs_va = fuzzy(fva[0].to(device)).cpu(); ql_va = refl(fva[1].to(device)).cpu()

    ths = cfg.thresholds
    combos = {"A": (ths.fd001_a if args.subset == "FD001" else ths.fd003_a),
              "B": (ths.fd001_b if args.subset == "FD001" else ths.fd003_b),
              "C": (ths.fd001_c if args.subset == "FD001" else ths.fd003_c)}
    print(f"[{args.subset}] 反思{'LN' if args.ln else '无LN'}（α={alpha} {cfg.fuzzy.pool_mode}）")
    for name, (t1, t2) in combos.items():
        yfv, nv, rv = combine(fva[2], fva[3], qs_va, ql_va, t1, t2)
        yft, nt, rt = combine(fte[2], fte[3], qs_te, ql_te, t1, t2)
        print(f"  {name}: val {rmse_mae(yfv, fva[4])[0]:.3f} | test {rmse_mae(yft, fte[4])[0]:.3f} "
              f"| lm {nt.float().mean()*100:.0f}% refl {rt.float().mean()*100:.0f}%")
    print(f"  Ql test 分位: [{ql_te.quantile(0.1):.2f}, {ql_te.median():.2f}, {ql_te.quantile(0.9):.2f}]")
    # 反思正确率（C 配置，论文表 III 口径）
    yt = fte[4]; ys = fte[2]; yl = fte[3]
    t1, t2 = combos["C"]
    need = qs_te < t1
    refl_m = need & (qs_te - ql_te > t2)
    if refl_m.sum():
        corr = ((ys[refl_m] - yt[refl_m]).abs() < (yl[refl_m] - yt[refl_m]).abs()).float().mean()
        print(f"  反思正确率(C) {corr*100:.1f}%（n={refl_m.sum()}）")


if __name__ == "__main__":
    main()

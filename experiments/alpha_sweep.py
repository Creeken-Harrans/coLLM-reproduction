"""阶段3 α 扫描：重训 FNN + 自反思（给定 α），评估组合 RMSE 与触发率。

目的：验证 α 对路由行为（触发率/反思率）与组合 RMSE 的权衡，对照论文
FLOPs 锚点（FD001 A/B/C = 25.3/47.6/78.9%）与 Fig 5 置信度 CDF（α≈10）。

复用 train_conf.py 的确定性训练配方（num_workers=0，固定 epochs，train+val 数据），
仅改变 α，其余冻结。SM/LM 权重直接用现有 checkpoint。

用法: python experiments/alpha_sweep.py --subset FD001 --alphas 6 8 10 12 15
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from collm.config import get_config
from collm.data import prepare_cmapss
from collm.models.collm import CoLLM
from collm.models.fuzzy import FuzzyAgent, confidence_label
from collm.models.reflection import ReflectionModel
from collm.train_common import set_seed


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


@torch.no_grad()
def extract(model, loader, device, alpha):
    fs, fl, qs_t, ql_t, ys, yl, yt = [], [], [], [], [], [], []
    model.eval()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        ys_b, feat_s = model.small(x)
        yl_b, feat_l = model.large(x)
        fs.append(feat_s.cpu()); fl.append(feat_l.cpu())
        qs_t.append(confidence_label(ys_b, y, alpha).cpu())
        ql_t.append(confidence_label(yl_b, y, alpha).cpu())
        ys.append(ys_b.cpu()); yl.append(yl_b.cpu()); yt.append(y.cpu())
    return (torch.cat(fs), torch.cat(fl), torch.cat(qs_t), torch.cat(ql_t),
            torch.cat(ys), torch.cat(yl), torch.cat(yt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--alphas", type=float, nargs="+", default=[6, 8, 10, 12, 15])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = get_config(args.subset)
    set_seed(cfg.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    ckpt_dir = Path(cfg.out_dir) / "checkpoints" / args.subset

    model = CoLLM(cfg).to(device)
    model.small.load_state_dict(torch.load(ckpt_dir / "small.pt", map_location=device))
    model.large.load_state_dict(torch.load(ckpt_dir / "large.pt", map_location=device))
    for m in (model.small, model.large):
        for p in m.parameters():
            p.requires_grad = False
    model.eval()

    tr_ds, val_ds, te_ds, _ = prepare_cmapss(cfg.data, args.subset, cfg.seed)
    feats = {}
    for name, ds in [("train", tr_ds), ("val", val_ds), ("test", te_ds)]:
        ld = DataLoader(ds, batch_size=cfg.train.conf_batch, shuffle=False,
                        num_workers=0, pin_memory=True)
        feats[name] = extract(model, ld, device, 6.0)  # α 仅用于标签，下面各自重算

    # 全样本推理结果（SM/LM 预测与特征，与 α 无关）
    ys_te, yl_te = feats["test"][4].numpy(), feats["test"][5].numpy()
    yt_te = feats["test"][6].numpy()
    print(f"[{args.subset}] SM RMSE {rmse(ys_te, yt_te):.3f} | LM RMSE {rmse(yl_te, yt_te):.3f}")

    ths = {"A": cfg.thresholds.fd001_a if args.subset == "FD001" else cfg.thresholds.fd003_a,
           "B": cfg.thresholds.fd001_b if args.subset == "FD001" else cfg.thresholds.fd003_b,
           "C": cfg.thresholds.fd001_c if args.subset == "FD001" else cfg.thresholds.fd003_c}

    print(f"\n{'α':>4} | {'A':>18} {'B':>18} {'C':>18} | 触发率A/B/C | 反思A/B/C")
    for alpha in args.alphas:
        # 重训 FNN + 反思（α）
        init_mu, init_sigma = FuzzyAgent.init_from_features(
            feats["train"][0], cfg.fuzzy.n_membership // cfg.fuzzy.d_input)
        fuzzy = FuzzyAgent(cfg.fuzzy, init_mu=init_mu, init_sigma=init_sigma).to(device)
        reflection = ReflectionModel(cfg.reflection).to(device)
        opt = torch.optim.Adam(list(fuzzy.parameters()) + list(reflection.parameters()),
                               lr=cfg.train.conf_lr)
        crit = nn.MSELoss()

        # 标签用当前 α 重算
        fs_tr, fl_tr = feats["train"][0], feats["train"][1]
        fsv, flv = feats["val"][0], feats["val"][1]
        ys_tr, yl_tr = feats["train"][4], feats["train"][5]
        yt_tr = feats["train"][6]
        ysv, ylv = feats["val"][4], feats["val"][5]
        ytv = feats["val"][6]
        qs_tr = confidence_label(ys_tr, yt_tr, alpha)
        ql_tr = confidence_label(yl_tr, yt_tr, alpha)
        qsv = confidence_label(ysv, ytv, alpha)
        qlv = confidence_label(ylv, ytv, alpha)
        ds = TensorDataset(torch.cat([fs_tr, fsv]), torch.cat([fl_tr, flv]),
                           torch.cat([qs_tr, qsv]), torch.cat([ql_tr, qlv]))
        ld = DataLoader(ds, batch_size=cfg.train.conf_batch, shuffle=True, num_workers=0)
        for _ in range(cfg.train.conf_epochs):
            fuzzy.train(); reflection.train()
            for a, b, c, d in ld:
                a, b, c, d = a.to(device), b.to(device), c.to(device), d.to(device)
                opt.zero_grad(set_to_none=True)
                loss = crit(fuzzy(a), c) + crit(reflection(b), d)
                loss.backward(); opt.step()

        # 测试推理 Qs/Ql
        fuzzy.eval(); reflection.eval()
        fs_te, fl_te = feats["test"][0].to(device), feats["test"][1].to(device)
        qs = fuzzy(fs_te).cpu().numpy()
        ql = reflection(fl_te).cpu().numpy()

        rows = []
        for name in ("A", "B", "C"):
            tau1, tau2 = ths[name]
            sm_exit = qs >= tau1
            need_lm = ~sm_exit
            delta = qs - ql
            refl = need_lm & (delta > tau2)
            yf = ys_te.copy()
            yf[need_lm] = np.where(refl[need_lm], (ys_te[need_lm] + yl_te[need_lm]) / 2,
                                   yl_te[need_lm])
            rows.append(f"{rmse(yf, yt_te):.3f}")
            trig = need_lm.mean() * 100
            refl_r = refl.mean() * 100
            if name == "A":
                trigs, refls = [], []
            trigs.append(trig); refls.append(refl_r)
        print(f"{alpha:>4} | {rows[0]:>6} {rows[1]:>6} {rows[2]:>6} | "
              f"{trigs[0]:.0f}/{trigs[1]:.0f}/{trigs[2]:.0f} | "
              f"{refls[0]:.0f}/{refls[1]:.0f}/{refls[2]:.0f}", flush=True)


if __name__ == "__main__":
    main()

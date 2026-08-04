"""CoLLM 评估与论文标准验收。

一次性对测试集全样本推理（SM+LM 全跑，用于完整统计），再按阈值组合:
  y_final = ys           若 Q_s ≥ τ1
          = yl           若 Q_s < τ1 且 Δ=Q_s−Q_l ≤ τ2
          = (ys+yl)/2    若 Q_s < τ1 且 Δ > τ2
输出: RMSE/MAE（CoLLM-A/B/C）、路由统计、FLOPs 加速比、自反思消融、
反思样本正确率（>70% SM 更优）、置信度分组 RMSE（图 5 风格）。

用法: python scripts/evaluate.py --subset FD001 [--alpha 15]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from collm.config import Config
from collm.data import prepare_cmapss
from collm.models.collm import CoLLM
from collm.flops import collm_flops, collm_speedup
from collm.train_common import set_seed


@torch.no_grad()
def full_inference(model: CoLLM, loader: DataLoader, device: str, batch: int = 512):
    """全样本推理，返回所有中间量（numpy）。"""
    ys_l, yl_l, qs_l, ql_l, yt_l = [], [], [], [], []
    model.eval()
    for x, y in loader:
        x = x.to(device)
        ys_b, feat_s = model.small(x)
        yl_b, feat_l = model.large(x)
        qs_b = model.fuzzy(feat_s, ys_b if model.cfg.fuzzy.cat_pred else None)
        ql_b = model.reflection(feat_l, yl_b if model.cfg.reflection.cat_pred else None)
        ys_l.append(ys_b.cpu()); yl_l.append(yl_b.cpu())
        qs_l.append(qs_b.cpu()); ql_l.append(ql_b.cpu())
        yt_l.append(y)
    return (torch.cat(ys_l).numpy(), torch.cat(yl_l).numpy(),
            torch.cat(qs_l).numpy(), torch.cat(ql_l).numpy(), torch.cat(yt_l).numpy())


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def combine(ys, yl, qs, ql, yt, tau1, tau2, use_reflection=True):
    """按论文规则组合最终预测。返回 (y_final, sm_exit, need_lm, reflect 全量掩码)。"""
    yf = ys.copy()
    sm_exit = qs >= tau1
    need_lm = ~sm_exit
    reflect = np.zeros_like(sm_exit)
    if use_reflection and need_lm.any():
        delta = qs[need_lm] - ql[need_lm]
        refl_sub = delta > tau2                       # 子集掩码
        reflect[need_lm] = refl_sub
        yf[need_lm] = np.where(refl_sub, (ys[need_lm] + yl[need_lm]) / 2.0, yl[need_lm])
    elif need_lm.any():
        yf[need_lm] = yl[need_lm]
    return yf, sm_exit, need_lm, reflect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001", choices=["FD001", "FD003"])
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--json", type=str, default=None, help="结果 JSON 输出路径")
    ap.add_argument("--feat-mode", default=None, choices=["fuzzy", "raw"])
    ap.add_argument("--pool-mode", default=None, choices=["mean", "stats", "flatten"])
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--blocks", type=int, default=None, help="LM 层数（FD001=9、FD003=12）")
    ap.add_argument("--ref-norm", action="store_true", help="反思网络输入 LayerNorm（阶段3 无 LN 版默认关）")
    ap.add_argument("--cat-pred", action="store_true", help="FNN/反思输入拼接预测值 ys/yl（联合分布实验）")
    args = ap.parse_args()

    cfg = Config()
    set_seed(cfg.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    alpha = args.alpha or cfg.fuzzy.alpha
    if args.feat_mode: cfg.fuzzy.feat_mode = args.feat_mode
    if args.pool_mode: cfg.fuzzy.pool_mode = args.pool_mode
    if args.hidden is not None: cfg.fuzzy.hidden = args.hidden
    if args.blocks is not None: cfg.large.n_blocks = args.blocks
    cfg.reflection.use_norm = args.ref_norm
    cfg.fuzzy.cat_pred = args.cat_pred
    cfg.reflection.cat_pred = args.cat_pred
    cfg.fuzzy.cat_pred = args.cat_pred
    cfg.reflection.cat_pred = args.cat_pred
    n_patch = (cfg.data.window + cfg.large.patch_stride) // cfg.large.patch_size if cfg.large.pad_patches else cfg.data.window // cfg.large.patch_size
    cfg.reflection.d_input = n_patch * cfg.large.d_embed
    ckpt_dir = Path(cfg.out_dir) / "checkpoints" / args.subset
    # 从权重自动推断 SM/LM 层数（防配置与权重不一致）
    _sd_sm = torch.load(ckpt_dir / "small.pt", map_location="cpu")
    cfg.small.n_layers = max(int(k.split(".")[2]) for k in _sd_sm if "encoder.layers." in k) + 1
    _sd_lg = torch.load(ckpt_dir / "large.pt", map_location="cpu")
    cfg.large.n_blocks = max(int(k.split(".")[2]) for k in _sd_lg if "gpt2.h." in k) + 1

    model = CoLLM(cfg).to(device)
    model.small.load_state_dict(torch.load(ckpt_dir / "small.pt", map_location=device))
    model.large.load_state_dict(torch.load(ckpt_dir / "large.pt", map_location=device))
    model.fuzzy.load_state_dict(torch.load(ckpt_dir / "fuzzy.pt", map_location=device))
    model.reflection.load_state_dict(torch.load(ckpt_dir / "reflection.pt", map_location=device))
    model.eval()

    tr_ds, val_ds, te_ds, stats = prepare_cmapss(cfg.data, args.subset, cfg.seed)
    te_ld = DataLoader(te_ds, batch_size=args.batch, shuffle=False)
    ys, yl, qs, ql, yt = full_inference(model, te_ld, device, args.batch)
    print(f"[{args.subset}] 测试样本 {len(ys)}")

    # ---- FLOPs ----
    x0 = te_ds[0][0].unsqueeze(0).to(device)
    flops = collm_flops(model, x0)
    for k, v in flops.items():
        print(f"  FLOPs[{k}] = {v/1e6:.2f}M")

    # ---- 阈值组合（论文正文 A/B/C + 表 III 的 [0.9, 0.05] 对照）----
    ths = cfg.thresholds
    combos = {
        "A": (ths.fd001_a if args.subset == "FD001" else ths.fd003_a),
        "B": (ths.fd001_b if args.subset == "FD001" else ths.fd003_b),
        "C": (ths.fd001_c if args.subset == "FD001" else ths.fd003_c),
        "T3-09": (0.9, 0.05),   # 表 III 消融所用阈值
    }
    results = {"subset": args.subset, "alpha": alpha, "n_test": len(ys),
               "baselines": {"SM": {"RMSE": rmse(ys, yt), "MAE": mae(ys, yt)},
                             "LM": {"RMSE": rmse(yl, yt), "MAE": mae(yl, yt)}},
               "flops": flops, "combos": {}, "ablation": {}}

    for name, (tau1, tau2) in combos.items():
        yf, sm_exit, need_lm, reflect = combine(ys, yl, qs, ql, yt, tau1, tau2)
        lm_ratio = need_lm.mean()
        speedup = collm_speedup(flops, lm_ratio)
        # 反思正确率：反思样本中 SM 误差更小（>70% 为论文标准；空切片置 0）
        refl_correct = 0.0
        if reflect.sum() > 0:
            refl_correct = float(np.mean(np.abs(ys[reflect] - yt[reflect])
                                         < np.abs(yl[reflect] - yt[reflect])))
        # 无自反思消融（论文：删除反思后指标变差 0.3%~1.1%）
        yf_nr, _, _, _ = combine(ys, yl, qs, ql, yt, tau1, tau2, use_reflection=False)
        entry = {
            "tau1": tau1, "tau2": tau2,
            "RMSE": rmse(yf, yt), "MAE": mae(yf, yt),
            # 论文“准确率”= 100×(1−(CoLLM−LM)/LM)：CoLLM 优于 LM 时 >100%
            "accuracy_vs_LM_RMSE": 100 * (2 - rmse(yf, yt) / rmse(yl, yt)),
            "accuracy_vs_LM_MAE": 100 * (2 - mae(yf, yt) / mae(yl, yt)),
            "sm_exit_ratio": float(sm_exit.mean()), "lm_ratio": float(lm_ratio),
            "reflect_ratio": float(reflect.mean()), "reflect_correct": refl_correct,
            "speedup": speedup,
            "no_reflection_RMSE": rmse(yf_nr, yt), "no_reflection_MAE": mae(yf_nr, yt),
        }
        results["combos"][name] = entry
        print(f"\n=== CoLLM-{name} [{tau1}, {tau2}] ===")
        print(f"  RMSE {entry['RMSE']:.3f} | MAE {entry['MAE']:.3f} "
              f"| 相对 LM 准确率 {entry['accuracy_vs_LM_RMSE']:.2f}%/{entry['accuracy_vs_LM_MAE']:.2f}%")
        print(f"  路由: SM 直接退出 {entry['sm_exit_ratio']*100:.1f}% | "
              f"大模型触发 {entry['lm_ratio']*100:.1f}% | 反思 {entry['reflect_ratio']*100:.1f}%")
        print(f"  反思正确率（SM 更优）{entry['reflect_correct']*100:.1f}% | "
              f"FLOPs 加速 {entry['speedup']:.2f}×")
        print(f"  消融(无自反思): RMSE {entry['no_reflection_RMSE']:.3f} MAE {entry['no_reflection_MAE']:.3f}")

    # ---- 置信度分组 RMSE（图 5：论文"ten confidence intervals"，等宽 0.0–1.0）----
    def bin_rmse(conf, err_abs):
        rows = []
        for k in range(10):
            lo, hi = 0.1 * k, 0.1 * (k + 1)
            idx = (conf >= lo) & (conf < hi) if k < 9 else (conf >= lo)  # 末箱含 1.0
            rows.append({"lo": lo, "hi": hi, "n": int(idx.sum()),
                         "rmse": float(np.sqrt(np.mean(err_abs[idx] ** 2))) if idx.sum() else None})
        return rows

    def monotonicity(rows):
        """置信度有效性的单调性指标：非空区间的 RMSE 与区间序号的 Spearman。"""
        xs, ys_ = [], []
        for k, r in enumerate(rows):
            if r["rmse"] is not None:
                xs.append(k); ys_.append(r["rmse"])
        return float(np.corrcoef(xs, ys_)[0, 1]) if len(xs) > 2 else float("nan")

    results["confidence_bins"] = {
        "SM": bin_rmse(qs, ys - yt),
        "LM": bin_rmse(ql, yl - yt),
    }
    results["confidence_monotonicity"] = {
        "SM": monotonicity(results["confidence_bins"]["SM"]),
        "LM": monotonicity(results["confidence_bins"]["LM"]),
    }
    print("\n=== 置信度分组 RMSE（等宽 0.0–1.0 十区间，图5）===")
    for who, rows in results["confidence_bins"].items():
        print(f"  {who}: " + "  ".join(f"{r['rmse'] if r['rmse'] is not None else float('nan'):.1f}" for r in rows)
              + f" | 单调性(负相关=有效) {results['confidence_monotonicity'][who]:+.3f}")

    out_path = args.json or (Path(cfg.out_dir) / "results" / f"{args.subset}_results.json")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()

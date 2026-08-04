"""论文图 3-6 风格可视化（中文校订版图号）。

- 图3: RUL 预测曲线 + 不确定度(Qs)与真实误差柱状图（4 个发动机）
- 图4: 自反思可视化：SM/LM 预测 + Error(LM−SM) 柱状图
- 图5: 置信度区间 RMSE 与样本分布（SM 与 LM）
- 图6: LM vs SM 预测曲线与误差对比（4 个发动机）

用法: python scripts/plot_results.py --subset FD001 [--threshold 0.6 0.05]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from collm.config import Config
from collm.data import prepare_cmapss
from collm.models.collm import CoLLM

# 中文字体（Noto Sans CJK SC，思源黑体）——DejaVu 不支持中文会乱码
plt.rcParams.update({
    "font.size": 9.5, "font.family": "Noto Sans CJK SC",
    "axes.unicode_minus": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "grid.color": "#cccccc",
    "axes.edgecolor": "#555555", "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "figure.dpi": 150, "savefig.dpi": 150,
    "legend.frameon": True, "legend.framealpha": 0.9, "legend.edgecolor": "#bbbbbb",
})

# 统一配色（论文风格：黑=真实、蓝=SM、红橙=LM、绿=不确定性、灰=误差）
C_TRUE = "#111111"
C_SM = "#1f6fb4"
C_LM = "#e4572e"
C_UNC = "#2a9d8f"
C_ERR = "#8d99ae"
C_ERR2 = "#e9a34d"
LW_TRUE, LW_PRED = 1.8, 1.2


def load_engine_windows(subset: str, engine_idx: int, cfg: Config, stats=None):
    """返回该发动机所有窗口的 (x, y_true)，x: (n, 50, 14)。

    stats: prepare_cmapss 的 (mu, sigma)——与正式评估管线完全一致的统计量
    （train-only 划分，防标准化不一致导致的曲线偏差）。
    """
    import numpy as np
    from collm.data import _load_raw, _extract_sensors, _unit_blocks, _build_windows, _subset_num
    te = _load_raw(cfg.data.data_dir, "test", subset)
    sens = _extract_sensors(te)
    units = te[:, 0].astype(np.int64)
    cycles = te[:, 1].astype(np.int64)
    blocks = _unit_blocks(units, cycles)
    block = blocks[engine_idx]
    x_w, end_cyc, _ = _build_windows(sens, [block], cfg.data.window, cfg.data.stride)
    if stats is None:
        # 兜底：与正式管线一致（prepare_cmapss 内部 train-only 统计）
        _, _, _, stats = prepare_cmapss(cfg.data, subset, cfg.seed)
    mu, sigma = stats["mu"], stats["sigma"]
    x_w = (x_w - mu) / np.where(sigma < 1e-8, 1.0, sigma)
    n_cyc = len(block[1])
    final_rul = np.loadtxt(Path(cfg.data.data_dir) / f"RUL_FD{_subset_num(subset)}.txt")[engine_idx]
    y_true = final_rul + (n_cyc - end_cyc)
    if cfg.data.rul_cap is not None:
        y_true = np.minimum(y_true, cfg.data.rul_cap)
    return x_w, y_true


def _pick_engines(subset: str, cfg: Config, want: int = 4, min_len: int = 120):
    """从测试集中挑选窗口数 ≥ min_len 的发动机（论文图 3/6 显示 ~160 时间步）。"""
    import numpy as np
    from collm.data import _load_raw, _extract_sensors, _unit_blocks, _build_windows
    te = _load_raw(cfg.data.data_dir, "test", subset)
    sens = _extract_sensors(te)
    units = te[:, 0].astype(np.int64)
    cycles = te[:, 1].astype(np.int64)
    blocks = _unit_blocks(units, cycles)
    picks = []
    for i, b in enumerate(blocks):
        if len(b[1]) >= min_len + cfg.data.window:
            picks.append(i)
        if len(picks) >= want:
            break
    return picks


def plot_fig3_6(model: CoLLM, subset: str, cfg: Config, device: str, out_dir: Path,
                stats=None):
    engines = _pick_engines(subset, cfg)
    print(f"选中发动机（0-based，长序列）: {engines}")
    # 布局：2×2 发动机网格，每格上下双面板（曲线 + 柱状），画布加大防拥挤
    fig3 = plt.figure(figsize=(15, 11.5))
    sub3 = fig3.subfigures(2, 2, hspace=0.32, wspace=0.10)
    fig6 = plt.figure(figsize=(15, 11.5))
    sub6 = fig6.subfigures(2, 2, hspace=0.32, wspace=0.10)
    for k, ei in enumerate(engines):
        x, yt = load_engine_windows(subset, ei, cfg, stats)
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
        with torch.no_grad():
            ys, fs = model.small(x)
            yl, fl = model.large(x)
            qs = model.fuzzy(fs)
            ql = model.reflection(fl)
        ys, yl, qs, ql, yt = [v.cpu().numpy() for v in (ys, yl, qs, ql)] + [yt]
        err_s, err_l = np.abs(ys - yt), np.abs(yl - yt)
        t = np.arange(len(yt))
        unc = 1 - qs   # 模型不确定性（FNN 置信度取反，对齐图 3 绿柱）

        # ---- 图 3：RUL 曲线 + FNN 不确定性 vs SM 真实误差 ----
        sf = sub3.flat[k]
        ax1 = sf.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [2, 1],
                                                          "hspace": 0.08})
        ax1[0].plot(t, yt, color=C_TRUE, lw=LW_TRUE, label="真实 RUL")
        ax1[0].plot(t, ys, color=C_SM, lw=LW_PRED, label="SM 预测")
        ax1[0].plot(t, yl, color=C_LM, lw=LW_PRED, label="LM 预测")
        ax1[0].set_ylabel("RUL")
        ax1[0].legend(fontsize=7.5, ncol=3, loc="upper right")
        ax1[0].set_title(f"发动机 #{ei+1}", fontsize=10, pad=4)
        # 柱状图：窄柱无描边，降低拥挤感
        ax1[1].bar(t, unc, color=C_UNC, alpha=0.7, width=0.7, label="FNN 不确定性 (1−Qs)")
        ax1[1].bar(t, err_s, color=C_ERR, alpha=0.5, width=0.7, label="SM 真实误差")
        ax1[1].set_ylabel("数值"); ax1[1].set_xlabel("时间步")
        ax1[1].legend(fontsize=7.5, ncol=2, loc="upper left")

        # ---- 图 6：RUL 曲线 + 误差柱（灰=LM、橙=SM）----
        sf6 = sub6.flat[k]
        ax2 = sf6.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [2, 1],
                                                          "hspace": 0.08})
        ax2[0].plot(t, yt, color=C_TRUE, lw=LW_TRUE, label="真实 RUL")
        ax2[0].plot(t, ys, color=C_SM, lw=LW_PRED, label="SM 预测")
        ax2[0].plot(t, yl, color="#2a9d8f", lw=LW_PRED, label="LM 预测")
        ax2[0].set_ylabel("RUL")
        ax2[0].legend(fontsize=7.5, ncol=3, loc="upper right")
        ax2[0].set_title(f"发动机 #{ei+1}", fontsize=10, pad=4)
        ax2[1].bar(t, err_l, color=C_ERR, alpha=0.55, width=0.7, label="LM 误差")
        ax2[1].bar(t, err_s, color=C_LM, alpha=0.5, width=0.7, label="SM 误差")
        ax2[1].set_ylabel("误差"); ax2[1].set_xlabel("时间步")
        ax2[1].legend(fontsize=7.5, ncol=2, loc="upper left")
    fig3.suptitle(f"图3 RUL 预测结果可视化（{subset}，上方曲线 / 下方 FNN 不确定性与 SM 真实误差）",
                  fontsize=12, y=0.99)
    fig6.suptitle(f"图6 大模型与小模型的 RUL 预测及误差比较（{subset}）",
                  fontsize=12, y=0.99)
    fig3.tight_layout(rect=[0, 0, 1, 0.97]); fig6.tight_layout(rect=[0, 0, 1, 0.97])
    fig3.savefig(out_dir / f"{subset}_fig3_rul_uncertainty.png")
    fig6.savefig(out_dir / f"{subset}_fig6_lm_vs_sm.png")
    plt.close(fig3); plt.close(fig6)
    print(f"图3/图6 已保存 → {out_dir}")


def plot_fig4(model: CoLLM, subset: str, cfg: Config, device: str, out_dir: Path,
              tau1: float = 0.9, tau2: float = 0.05, stats=None):
    """图4：自反思可视化——真实 RUL、SM/LM 预测 + Error(LM−SM) 柱状图。

    红柱 = LM 误差 > SM 误差（自反思应触发融合）；蓝柱 = LM 更优。
    采用表 III 的阈值 [0.9, 0.05] 以突出反思样本。
    """
    engine = _pick_engines(subset, cfg, want=1)[0]  # 用长序列发动机展示
    x, yt = load_engine_windows(subset, engine, cfg, stats)
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    with torch.no_grad():
        ys, fs = model.small(x)
        yl, fl = model.large(x)
        qs, ql = model.fuzzy(fs), model.reflection(fl)
    ys, yl, qs, ql, yt = [v.cpu().numpy() for v in (ys, yl, qs, ql)] + [yt]

    err_s, err_l = np.abs(ys - yt), np.abs(yl - yt)
    delta = qs - ql
    reflect = (qs < tau1) & (delta > tau2)   # 触发自反思融合的样本
    t = np.arange(len(yt))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2.2, 1]})
    ax1.plot(t, yt, color=C_TRUE, lw=LW_TRUE, label="真实 RUL")
    ax1.plot(t, ys, color=C_SM, lw=LW_PRED, label="SM 预测")
    ax1.plot(t, yl, color=C_LM, lw=LW_PRED, label="LM 预测")
    ax1.scatter(t[reflect], ys[reflect], c="#d62828", s=18, zorder=5,
                label=f"自反思触发({reflect.sum()} 个)")
    ax1.set_ylabel("RUL"); ax1.legend(fontsize=8); ax1.set_title(f"图4 自反思（{subset} #{engine+1}，阈值 [{tau1},{tau2}]）")

    colors = np.where(err_l > err_s, "#d1495b", "#2f6db3")   # 红=LM 更差，蓝=LM 更优
    ax2.bar(t, err_l - err_s, color=colors, width=0.9)
    ax2.axhline(0, color=C_TRUE, lw=0.8)
    ax2.set_xlabel("时间步"); ax2.set_ylabel("Error(LM−SM)")
    fig.tight_layout()
    fig.savefig(out_dir / f"{subset}_fig4_self_reflection.png")
    plt.close(fig)
    print(f"图4 已保存 → {out_dir}（反思触发 {reflect.sum()} 样本）")


def plot_fig5(model: CoLLM, te_ld: DataLoader, device: str, out_dir: Path, subset: str):
    ys_l, yl_l, qs_l, ql_l, yt_l = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for x, y in te_ld:
            x = x.to(device)
            ys, fs = model.small(x); yl, fl = model.large(x)
            qs_l.append(model.fuzzy(fs).cpu()); ql_l.append(model.reflection(fl).cpu())
            ys_l.append(ys.cpu()); yl_l.append(yl.cpu()); yt_l.append(y)
    ys = torch.cat(ys_l).numpy(); yl = torch.cat(yl_l).numpy()
    qs = torch.cat(qs_l).numpy(); ql = torch.cat(ql_l).numpy()
    yt = torch.cat(yt_l).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (name, q, err) in zip(axes, [("SM", qs, ys - yt), ("LM", ql, yl - yt)]):
        # 论文：ten confidence intervals，x 轴 0.0–1.0 → 等宽区间 [0.1k, 0.1(k+1)]
        frac, rmse_bin, mid = [], [], []
        for k in range(10):
            lo, hi = 0.1 * k, 0.1 * (k + 1)
            idx = (q >= lo) & (q < hi)
            mid.append(lo + 0.05)
            frac.append(100 * idx.sum() / len(q))
            rmse_bin.append(np.sqrt(np.mean(err[idx] ** 2)) if idx.sum() else float("nan"))
        # 论文 Fig.5：左轴橙色柱=样本占比（%），右轴蓝色虚线=区间 RMSE
        ax.bar(mid, frac, width=0.085, color="#e9a34d", alpha=0.85, label="样本占比 %")
        ax2 = ax.twinx()
        ax2.plot(mid, rmse_bin, color=C_SM, ls="--", marker="o", ms=3.5, lw=1.4,
                 label="区间 RMSE")
        ax.set_title(f"图5 {name} 置信度")
        ax.set_xlabel("置信度"); ax.set_ylabel("样本占比(%)", color="#e9a34d")
        ax2.set_ylabel("RMSE", color=C_SM)
        ax2.set_ylim(bottom=0)
        ax2.legend(loc="upper right", fontsize=8)
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{subset}_fig5_confidence_bins.png")
    plt.close(fig)
    print(f"图5 已保存 → {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001", choices=["FD001", "FD003"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pool-mode", default=None, choices=["mean", "stats"],
                    help="FNN 时间聚合（FD001=stats、FD003=mean，对齐最终配置）")
    ap.add_argument("--blocks", type=int, default=None, help="LM 层数（FD001=9、FD003=12）")
    ap.add_argument("--hidden", type=int, default=None, help="FNN hidden（0=单层，匹配阶段3 最终配置）")
    ap.add_argument("--ref-norm", action="store_true", help="反思网络 LayerNorm（阶段3 无 LN 默认关）")
    args = ap.parse_args()

    cfg = Config()
    if args.pool_mode: cfg.fuzzy.pool_mode = args.pool_mode
    if args.blocks is not None: cfg.large.n_blocks = args.blocks
    if args.hidden is not None: cfg.fuzzy.hidden = args.hidden
    cfg.reflection.use_norm = args.ref_norm
    device = args.device if torch.cuda.is_available() else "cpu"
    ckpt = Path(cfg.out_dir) / "checkpoints" / args.subset
    # 从权重自动推断 SM/LM 层数（与 train_conf/evaluate 一致）
    _sd = torch.load(ckpt / "small.pt", map_location="cpu")
    cfg.small.n_layers = max(int(k.split(".")[2]) for k in _sd if "encoder.layers." in k) + 1
    _sd2 = torch.load(ckpt / "large.pt", map_location="cpu")
    cfg.large.n_blocks = max(int(k.split(".")[2]) for k in _sd2 if "gpt2.h." in k) + 1
    model = CoLLM(cfg).to(device).eval()
    for part, name in [("small", "small"), ("large", "large"),
                       ("fuzzy", "fuzzy"), ("reflection", "reflection")]:
        model.__getattr__(part).load_state_dict(
            torch.load(ckpt / f"{name}.pt", map_location=device))

    out_dir = Path(cfg.out_dir) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    # 标准化统计量一次性获取（与正式评估管线一致，供图 3/4/6 复用）
    _, _, te_ds, stats = prepare_cmapss(cfg.data, args.subset, cfg.seed)
    plot_fig3_6(model, args.subset, cfg, device, out_dir, stats)
    plot_fig4(model, args.subset, cfg, device, out_dir, stats=stats)
    te_ld = DataLoader(te_ds, batch_size=512, shuffle=False)
    plot_fig5(model, te_ld, device, out_dir, args.subset)
    print("全部图表完成")


if __name__ == "__main__":
    main()

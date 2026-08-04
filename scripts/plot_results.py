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

from collm.config import get_config
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


def load_engine_windows(subset: str, engine_idx: int, cfg, stats=None):
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


def _pick_engines(subset: str, cfg, want: int = 4, min_len: int = 90):
    """选择展示发动机：优先论文图 3/6 指定的编号（FD001 #32/#34、FD003 #23/#40，
    1-based → 0-based），不足时退回长序列发动机。"""
    import numpy as np
    from collm.data import _load_raw, _extract_sensors, _unit_blocks, _build_windows
    te = _load_raw(cfg.data.data_dir, "test", subset)
    units = te[:, 0].astype(np.int64)
    blocks = _unit_blocks(units, te[:, 1].astype(np.int64))
    paper_idx = {"FD001": [31, 33], "FD003": [22, 39]}   # 论文图 3/6 编号（0-based）
    picks = [i for i in paper_idx.get(subset, []) if i < len(blocks)
             and len(blocks[i][1]) >= min_len + cfg.data.window]
    # 论文编号不足时补充长序列发动机
    for i, b in enumerate(blocks):
        if len(picks) >= want:
            break
        if i not in picks and len(b[1]) >= min_len + cfg.data.window:
            picks.append(i)
    return picks[:want]


def plot_fig3_6(models: dict, cfgs: dict, device: str, out_dir: Path):
    """图3/图6（论文布局）：4 面板 = (a)FD001 #32 (b)FD001 #34 (c)FD003 #23 (d)FD003 #40。

    图3 每面板：上=RUL 曲线（真实黑、SM 蓝、LM 绿——论文 Fig.6 配色），
              下=绿色不确定性柱（FNN 1−Qs）+ 灰色 SM 真实误差柱（论文 Fig.3 原文）。
    图6 每面板：上=曲线，下=灰色 LM 误差柱 + 蓝色 SM 误差柱（论文 Fig.6 原文）。
    """
    panels = [("FD001", 31), ("FD001", 33), ("FD003", 22), ("FD003", 39)]
    fig3 = plt.figure(figsize=(15, 11.5))
    sub3 = fig3.subfigures(2, 2, hspace=0.30, wspace=0.10)
    fig6 = plt.figure(figsize=(15, 11.5))
    sub6 = fig6.subfigures(2, 2, hspace=0.30, wspace=0.10)
    for k, (sub, ei) in enumerate(panels):
        cfg = cfgs[sub]; model = models[sub]
        _, _, _, stats = prepare_cmapss(cfg.data, sub, cfg.seed)
        x, yt = load_engine_windows(sub, ei, cfg, stats)
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
        with torch.no_grad():
            ys, fs = model.small(x)
            yl, fl = model.large(x)
            qs = model.fuzzy(fs)
        ys, yl, qs, yt = [v.cpu().numpy() for v in (ys, yl, qs)] + [yt]
        err_s, err_l = np.abs(ys - yt), np.abs(yl - yt)
        t = np.arange(len(yt))
        unc = 1 - qs   # 模型不确定性（论文图 3 绿柱）

        # ---- 图 3 ----
        sf = sub3.flat[k]
        ax1 = sf.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [2, 1],
                                                          "hspace": 0.08})
        ax1[0].plot(t, yt, color=C_TRUE, lw=LW_TRUE, label="真实 RUL")
        ax1[0].plot(t, ys, color=C_SM, lw=LW_PRED, label="SM 预测")
        ax1[0].plot(t, yl, color="#2a9d8f", lw=LW_PRED, label="LM 预测")
        ax1[0].set_ylabel("RUL")
        ax1[0].legend(fontsize=7.5, ncol=3, loc="upper right")
        ax1[0].set_title(f"({chr(97+k)}) {sub} 发动机 #{ei+1}", fontsize=10, pad=4)
        ax1[1].bar(t, unc, color=C_UNC, alpha=0.7, width=0.7, label="模型不确定性 (1−Qs)")
        ax1[1].bar(t, err_s, color=C_ERR, alpha=0.5, width=0.7, label="SM 真实误差")
        ax1[1].set_ylabel("数值"); ax1[1].set_xlabel("时间步")
        ax1[1].legend(fontsize=7.5, ncol=2, loc="upper left")

        # ---- 图 6 ----
        sf6 = sub6.flat[k]
        ax2 = sf6.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [2, 1],
                                                          "hspace": 0.08})
        ax2[0].plot(t, yt, color=C_TRUE, lw=LW_TRUE, label="真实 RUL")
        ax2[0].plot(t, ys, color=C_SM, lw=LW_PRED, label="SM 预测")
        ax2[0].plot(t, yl, color="#2a9d8f", lw=LW_PRED, label="LM 预测")
        ax2[0].set_ylabel("RUL")
        ax2[0].legend(fontsize=7.5, ncol=3, loc="upper right")
        ax2[0].set_title(f"({chr(97+k)}) {sub} 发动机 #{ei+1}", fontsize=10, pad=4)
        ax2[1].bar(t, err_l, color=C_ERR, alpha=0.55, width=0.7, label="LM 误差")
        ax2[1].bar(t, err_s, color=C_SM, alpha=0.5, width=0.7, label="SM 误差")
        ax2[1].set_ylabel("误差"); ax2[1].set_xlabel("时间步")
        ax2[1].legend(fontsize=7.5, ncol=2, loc="upper left")
    fig3.suptitle("图3 RUL 预测结果可视化（上：预测曲线 / 下：FNN 不确定性与 SM 真实误差）",
                  fontsize=12, y=0.99)
    fig6.suptitle("图6 大模型与小模型的 RUL 预测及误差比较",
                  fontsize=12, y=0.99)
    fig3.tight_layout(rect=[0, 0, 1, 0.97]); fig6.tight_layout(rect=[0, 0, 1, 0.97])
    fig3.savefig(out_dir / "fig3_rul_uncertainty.png")
    fig6.savefig(out_dir / "fig6_lm_vs_sm.png")
    plt.close(fig3); plt.close(fig6)
    print("图3/图6 已保存（论文布局：FD001 #32/#34 + FD003 #23/#40）")


def plot_fig4(models: dict, cfgs: dict, device: str, out_dir: Path):
    """图4（论文布局）：(a) 阈值 [0.9,0.05] (b) [0.6,0.05]（表 III 消融阈值）。

    上=真实 RUL + SM/LM 预测；下=Error(LM−SM) 柱状图：
    红柱 = LM 误差 > SM 误差（自反思触发融合的样本），蓝柱 = LM 更优。
    展示发动机：FD001 #34（论文图 6 中 SM/LM 差异显著的案例）。
    """
    sub, ei = "FD001", 33   # 论文图 6 分析案例（SM 早期更准 / LM 后期更稳）
    cfg = cfgs[sub]; model = models[sub]
    _, _, _, stats = prepare_cmapss(cfg.data, sub, cfg.seed)
    x, yt = load_engine_windows(sub, ei, cfg, stats)
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    with torch.no_grad():
        ys, fs = model.small(x)
        yl, fl = model.large(x)
        qs, ql = model.fuzzy(fs), model.reflection(fl)
    ys, yl, qs, ql, yt = [v.cpu().numpy() for v in (ys, yl, qs, ql)] + [yt]
    err_s, err_l = np.abs(ys - yt), np.abs(yl - yt)
    t = np.arange(len(yt))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True,
                             gridspec_kw={"width_ratios": [2.2, 1], "hspace": 0.35,
                                          "wspace": 0.18})
    for col, (tau1, tau2) in enumerate([(0.9, 0.05), (0.6, 0.05)]):
        delta = qs - ql
        reflect = (qs < tau1) & (delta > tau2)   # 触发自反思融合的样本
        ax1, ax2 = axes[:, col]
        ax1.plot(t, yt, color=C_TRUE, lw=LW_TRUE, label="真实 RUL")
        ax1.plot(t, ys, color=C_SM, lw=LW_PRED, label="SM 预测")
        ax1.plot(t, yl, color="#2a9d8f", lw=LW_PRED, label="LM 预测")
        ax1.scatter(t[reflect], ys[reflect], c="#d62828", s=16, zorder=5,
                    label=f"自反思触发 ({reflect.sum()})")
        ax1.set_ylabel("RUL")
        ax1.legend(fontsize=7.5)
        ax1.set_title(f"({chr(97+col)}) 阈值 [{tau1}, {tau2}]", fontsize=10)
        colors = np.where(err_l > err_s, "#d1495b", "#2f6db3")   # 红=LM 更差，蓝=LM 更优
        ax2.bar(t, err_l - err_s, color=colors, width=0.9)
        ax2.axhline(0, color=C_TRUE, lw=0.8)
        ax2.set_xlabel("时间步"); ax2.set_ylabel("Error(LM−SM)")
    fig.suptitle(f"图4 自反思机制（{sub} 发动机 #{ei+1}，红柱=LM 误差更大→SM 辅助融合）",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "fig4_self_reflection.png")
    plt.close(fig)
    print("图4 已保存（双阈值 (a)[0.9,0.05] (b)[0.6,0.05]）")


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
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    out_dir = Path("outputs") / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载两个数据集的模型（图 3/6 面板跨数据集）
    models, cfgs = {}, {}
    for sub in ("FD001", "FD003"):
        cfg = get_config(sub)
        model = CoLLM(cfg).to(device).eval()
        ckpt = Path(cfg.out_dir) / "checkpoints" / sub
        for part, name in [("small", "small"), ("large", "large"),
                           ("fuzzy", "fuzzy"), ("reflection", "reflection")]:
            model.__getattr__(part).load_state_dict(
                torch.load(ckpt / f"{name}.pt", map_location=device))
        models[sub] = model
        cfgs[sub] = cfg

    plot_fig3_6(models, cfgs, device, out_dir)
    plot_fig4(models, cfgs, device, out_dir)
    # 图5：SM/LM 置信度分箱（橙柱+蓝虚线，论文 Fig.5）——两数据集分两张
    for sub in ("FD001", "FD003"):
        cfg = cfgs[sub]; model = models[sub]
        _, _, te_ds, _ = prepare_cmapss(cfg.data, sub, cfg.seed)
        te_ld = DataLoader(te_ds, batch_size=512, shuffle=False)
        plot_fig5(model, te_ld, device, out_dir, sub)
    print("全部图表完成（论文布局：图3/4/6 跨数据集各一张，图5 每数据集一张）")


if __name__ == "__main__":
    main()

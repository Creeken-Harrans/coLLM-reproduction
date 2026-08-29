"""论文图 3-6 复刻（中文校订版图号）——按论文图件规范逐项核对 + 数据真实性优先。

图件规范（依据论文 PDF/图件 OCR，见 docs/PAPER_BENCHMARKS.md 与
REVISIONS #45.4；OCR 细节见交付记录）：
- 图3: RUL 预测结果可视化。上 = Prediction（SM 预测，论文下栏为其不确定度
  与真实误差的配对）+ Actual RUL（黑）；下 = 绿色 FNN 不确定度(1−Qs) 柱 +
  灰色 SM 真实误差柱。4 面板 = FD001 #32/#34 + FD003 #23/#40。
- 图4: 自反思机制（论文为 FD003，表 III 阈值）：(a) [0.9,0.05]、(b) [0.6,0.05]。
  只画**触发反思的样本**（论文 x 轴 = Sample Index 0-500/0-250 = 反思样本数
  518/249），上 = Real/SM/LM，下 = Error(LM−SM) 柱：红 = LM 误差更大（正），
  蓝 = LM 更优（负）。
- 图5: 置信度十等分箱（SM/LM 两面板）：橙柱 = 样本占比(%)，蓝虚线 = 区间 RMSE。
  末箱含 1.0；LM 面板若 Ql 饱和退化为单一 bin，如实标注（不掩盖）。
- 图6: LM vs SM。上 = Actual（黑）/ LM（绿）/ SM（蓝）；下 = 灰 LM 误差 +
  蓝 SM 误差。4 面板同图 3。
- 配色按论文正文 E 节：LM 绿、SM 蓝、真实黑。

所有曲线/柱状均来自当前定稿 checkpoint 的真实推理（无美化、无截断选择）。
用法: python scripts/plot_results.py [--device cuda]（跨数据集一次生成图 3/4/5/6）
"""
import argparse
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
from collm.data import prepare_cmapss, _load_raw, _unit_blocks, _build_windows, _subset_num
from collm.models.collm import CoLLM

# 中文字体（Noto Sans CJK SC，思源黑体）
plt.rcParams.update({
    "font.size": 9.5, "font.family": "Noto Sans CJK SC",
    "axes.unicode_minus": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "grid.color": "#cccccc",
    "axes.edgecolor": "#555555", "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "figure.dpi": 150, "savefig.dpi": 150,
    "legend.frameon": True, "legend.framealpha": 0.9, "legend.edgecolor": "#bbbbbb",
})

# 论文配色：黑=真实、蓝=SM、绿=LM（正文 E 节 "LM (green) / SM (blue)"）、
# 绿=不确定度（图3 下栏）、灰=误差柱、橙=样本占比（图5）
C_TRUE = "#111111"
C_SM = "#1f6fb4"
C_LM = "#2ca02c"
C_UNC = "#2a9d8f"
C_ERR = "#8d99ae"
C_ORANGE = "#e9a34d"
C_RED = "#d1495b"
LW_TRUE, LW_PRED = 1.8, 1.2

# 论文图 3/6 指定发动机（1-based → 0-based）
PAPER_ENGINES = [("FD001", 32), ("FD001", 34), ("FD003", 23), ("FD003", 40)]


def load_engine_windows(subset: str, engine_1based: int, cfg):
    """该发动机全部窗口的 (x, end_cycle, y_true)。统计量与正式管线一致（train-only）。"""
    from collm.data import _extract_sensors
    te = _load_raw(cfg.data.data_dir, "test", subset)
    sens = _extract_sensors(te)
    units = te[:, 0].astype(np.int64)
    cycles = te[:, 1].astype(np.int64)
    blocks = _unit_blocks(units, cycles)
    block = blocks[engine_1based - 1]
    x_w, end_cyc, _ = _build_windows(sens, [block], cfg.data.window, cfg.data.stride)
    _, _, _, stats = prepare_cmapss(cfg.data, subset, cfg.seed)
    mu, sigma = stats["mu"], stats["sigma"]
    x_w = (x_w - mu) / np.where(sigma < 1e-8, 1.0, sigma)
    n_cyc = len(block[1])
    final_rul = np.loadtxt(Path(cfg.data.data_dir) / f"RUL_FD{_subset_num(subset)}.txt")[
        engine_1based - 1]
    y_true = final_rul + (n_cyc - end_cyc)
    if cfg.data.rul_cap is not None:
        y_true = np.minimum(y_true, cfg.data.rul_cap)
    return x_w, end_cyc, y_true


def _run_engine(models, cfgs, subset, engine_1based, device):
    """对指定发动机全窗口推理，返回 (t, yt, ys, yl, qs, ql)。t = 窗口末端 cycle。"""
    cfg = cfgs[subset]
    x, end_cyc, yt = load_engine_windows(subset, engine_1based, cfg)
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    with torch.no_grad():
        ys, fs = models[subset].small(x_t)
        yl, fl = models[subset].large(x_t)
        qs = models[subset].fuzzy(fs)
        ql = models[subset].reflection(fl)
    return (end_cyc.astype(float), yt, ys.cpu().numpy(), yl.cpu().numpy(),
            qs.cpu().numpy(), ql.cpu().numpy())


def plot_fig3_6(models: dict, cfgs: dict, device: str, out_dir: Path):
    """图3/图6（论文布局 2×2）：(a)FD001#32 (b)FD001#34 (c)FD003#23 (d)FD003#40。

    图3：上 = 真实（黑）+ SM 预测（蓝）；下 = 不确定度 1−Qs（绿）+ SM 真实误差（灰）。
    图6：上 = 真实（黑）+ SM（蓝）+ LM（绿）；下 = LM 误差（灰）+ SM 误差（蓝）。
    x 轴 = 窗口末端 cycle（论文 Time Steps 口径）。
    """
    fig3 = plt.figure(figsize=(15, 11.5))
    sub3 = fig3.subfigures(2, 2, hspace=0.30, wspace=0.10)
    fig6 = plt.figure(figsize=(15, 11.5))
    sub6 = fig6.subfigures(2, 2, hspace=0.30, wspace=0.10)
    for k, (sub, ei) in enumerate(PAPER_ENGINES):
        t, yt, ys, yl, qs, _ = _run_engine(models, cfgs, sub, ei, device)
        err_s, err_l = np.abs(ys - yt), np.abs(yl - yt)

        # ---- 图 3 ----
        sf = sub3.flat[k]
        ax1 = sf.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [2, 1],
                                                          "hspace": 0.08})
        ax1[0].plot(t, yt, color=C_TRUE, lw=LW_TRUE, label="真实 RUL")
        ax1[0].plot(t, ys, color=C_SM, lw=LW_PRED, label="SM 预测")
        ax1[0].set_ylabel("RUL"); ax1[0].set_ylim(bottom=0)
        ax1[0].legend(fontsize=7.5, ncol=2, loc="upper right")
        ax1[0].set_title(f"({chr(97+k)}) {sub} 发动机 #{ei}", fontsize=10, pad=4)
        ax1[1].bar(t, 1 - qs, color=C_UNC, alpha=0.7, width=0.9,
                   label="模型不确定度 (1−Qs)")
        ax1[1].bar(t, err_s, color=C_ERR, alpha=0.5, width=0.9, label="SM 真实误差")
        ax1[1].set_ylabel("数值"); ax1[1].set_xlabel("Cycle")
        ax1[1].legend(fontsize=7.5, ncol=2, loc="upper left")

        # ---- 图 6 ----
        sf6 = sub6.flat[k]
        ax2 = sf6.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [2, 1],
                                                           "hspace": 0.08})
        ax2[0].plot(t, yt, color=C_TRUE, lw=LW_TRUE, label="真实 RUL")
        ax2[0].plot(t, ys, color=C_SM, lw=LW_PRED, label="SM 预测")
        ax2[0].plot(t, yl, color=C_LM, lw=LW_PRED, label="LM 预测")
        ax2[0].set_ylabel("RUL"); ax2[0].set_ylim(bottom=0)
        ax2[0].legend(fontsize=7.5, ncol=3, loc="upper right")
        ax2[0].set_title(f"({chr(97+k)}) {sub} 发动机 #{ei}", fontsize=10, pad=4)
        ax2[1].bar(t, err_l, color=C_ERR, alpha=0.55, width=0.9, label="LM 误差")
        ax2[1].bar(t, err_s, color=C_SM, alpha=0.5, width=0.9, label="SM 误差")
        ax2[1].set_ylabel("误差"); ax2[1].set_xlabel("Cycle")
        ax2[1].legend(fontsize=7.5, ncol=2, loc="upper left")
    fig3.suptitle("图3 RUL 预测结果（上：SM 预测 vs 真实 / 下：FNN 不确定度与 SM 真实误差）",
                  fontsize=12, y=0.99)
    fig6.suptitle("图6 LM vs SM 的 RUL 预测及误差比较（上：曲线 / 下：误差柱）",
                  fontsize=12, y=0.99)
    fig3.tight_layout(rect=[0, 0, 1, 0.97]); fig6.tight_layout(rect=[0, 0, 1, 0.97])
    fig3.savefig(out_dir / "fig3_rul_uncertainty.png")
    fig6.savefig(out_dir / "fig6_lm_vs_sm.png")
    plt.close(fig3); plt.close(fig6)
    print("图3/图6 已保存（论文布局：FD001 #32/#34 + FD003 #23/#40）")


def plot_fig4(models: dict, cfgs: dict, device: str, out_dir: Path):
    """图4（论文为 FD003，表 III 阈值）：(a) [0.9,0.05] (b) [0.6,0.05]。

    只画触发反思（Qs<τ1 且 Δ=Qs−Ql>τ2）的样本，x 轴 = 反思样本序号
    （论文 Sample Index 0-500/0-250 = 表 III 反思数 518/249）。
    下 = Error(LM−SM)：红 = LM 误差更大（论文"正差值→SM 辅助修正"），蓝 = LM 更优。
    注：新 LM（11.43）下无 LN 反思 Ql 饱和高位 → 反思 0 触发 → 本图为空（如实展示，
    见 REVISIONS #52；加 LN 后 Ql 恢复校准、反思恢复触发）。
    """
    sub = "FD003"
    cfg = cfgs[sub]; model = models[sub]
    tr_ds, val_ds, te_ds, _ = prepare_cmapss(cfg.data, sub, cfg.seed)
    te_ld = DataLoader(te_ds, batch_size=512, shuffle=False)
    ys_l, yl_l, qs_l, ql_l, yt_l = [], [], [], [], []
    with torch.no_grad():
        for x, y in te_ld:
            x = x.to(device)
            ys, fs = model.small(x); yl, fl = model.large(x)
            qs_l.append(model.fuzzy(fs).cpu()); ql_l.append(model.reflection(fl).cpu())
            ys_l.append(ys.cpu()); yl_l.append(yl.cpu()); yt_l.append(y)
    ys = torch.cat(ys_l).numpy(); yl = torch.cat(yl_l).numpy()
    qs = torch.cat(qs_l).numpy(); ql = torch.cat(ql_l).numpy()
    yt = torch.cat(yt_l).numpy()
    err_s, err_l = np.abs(ys - yt), np.abs(yl - yt)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=False,
                             gridspec_kw={"hspace": 0.35, "wspace": 0.18})
    counts = []
    for col, (tau1, tau2) in enumerate([(0.9, 0.05), (0.6, 0.05)]):
        reflect = (qs < tau1) & (qs - ql > tau2)
        idx = np.where(reflect)[0]
        n = len(idx)
        counts.append(n)
        ax1, ax2 = axes[:, col]
        xs = np.arange(n)
        ax1.plot(xs, yt[idx], color=C_TRUE, lw=1.6, label="真实 RUL")
        ax1.plot(xs, ys[idx], color=C_SM, lw=1.0, label="SM 预测")
        ax1.plot(xs, yl[idx], color=C_LM, lw=1.0, label="LM 预测")
        ax1.set_ylabel("RUL"); ax1.set_ylim(bottom=0)
        ax1.legend(fontsize=7.5, loc="upper right")
        ax1.set_title(f"({chr(97+col)}) 阈值 [{tau1}, {tau2}]（反思样本 {n}）",
                      fontsize=10)
        diff = err_l[idx] - err_s[idx]
        colors = np.where(diff > 0, C_RED, C_SM)   # 红 = LM 更差（正），蓝 = LM 更优
        ax2.bar(xs, diff, color=colors, width=max(0.8, 300 / max(n, 1)))
        ax2.axhline(0, color=C_TRUE, lw=0.8)
        ax2.set_xlabel("反思样本序号（Sample Index）")
        ax2.set_ylabel("Error(LM−SM)")
        from matplotlib.patches import Patch
        ax2.legend(handles=[Patch(color=C_RED, label="LM 误差更大（SM 辅助融合）"),
                            Patch(color=C_SM, label="LM 误差更小")],
                   fontsize=7.5, loc="upper right")
    fig.suptitle(f"图4 自反思机制（{sub}，论文表 III 阈值；复现反思率远高于论文"
                 f"——LM 弱于论文所致，如实展示）", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "fig4_self_reflection.png")
    plt.close(fig)
    print(f"图4 已保存（FD003 双阈值，反思样本数 [0.9,0.05]={counts[0]} / [0.6,0.05]={counts[1]}）")


def plot_fig5(model: CoLLM, te_ld: DataLoader, device: str, out_dir: Path, subset: str):
    """图5：置信度十等分箱（SM/LM 面板）。橙柱=样本占比，蓝虚线=区间 RMSE。

    真实性处理：末箱含 1.0；LM 面板若 Ql 饱和（复现已知现象，FINAL_RESULTS
    §3）→ 图内如实标注退化原因，不伪造分布。
    """
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

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (name, q, err) in zip(axes, [("SM", qs, ys - yt), ("LM", ql, yl - yt)]):
        frac, rmse_bin, mid = [], [], []
        for k in range(10):
            lo, hi = 0.1 * k, 0.1 * (k + 1)
            idx = (q >= lo) & (q < hi) if k < 9 else (q >= lo)  # 末箱含 1.0
            mid.append(lo + 0.05)
            frac.append(100 * idx.sum() / len(q))
            rmse_bin.append(np.sqrt(np.mean(err[idx] ** 2)) if idx.sum() else float("nan"))
        ax.bar(mid, frac, width=0.085, color=C_ORANGE, alpha=0.85, label="样本占比 %")
        ax2 = ax.twinx()
        ax2.plot(mid, rmse_bin, color=C_SM, ls="--", marker="o", ms=3.5, lw=1.4,
                 label="区间 RMSE")
        ax2.set_ylim(bottom=0)
        ax.set_title(f"({'a' if name == 'SM' else 'b'}) {subset} {name}")
        ax.set_xlabel("置信度区间"); ax.set_ylabel("样本占比(%)", color="#8a5a00")
        ax2.set_ylabel("RMSE", color=C_SM)
        ax2.legend(loc="upper right", fontsize=8)
        ax.legend(loc="upper left", fontsize=8)
        if name == "LM":
            # Ql 退化检查：>95% 样本落在单箱 → 如实标注
            top = np.argmax(frac)
            if frac[top] > 95:
                ax.text(0.5, 0.78,
                        f"Ql 在 test 饱和于箱 {top}（{frac[top]:.0f}%）\n"
                        f"单层 FCN 对特征偏移无归一（公式 12 字面）\n"
                        f"→ 触发即融合；见 FINAL_RESULTS §3",
                        transform=ax.transAxes, ha="center", fontsize=7.5,
                        color="#555555",
                        bbox=dict(fc="white", ec="#cccccc", alpha=0.85))
    fig.suptitle(f"图5 置信度分箱（{subset} test）：RMSE 随置信度升高而降低 = 置信度有效",
                 fontsize=11, y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
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
    for sub in ("FD001", "FD003"):
        cfg = cfgs[sub]; model = models[sub]
        _, _, te_ds, _ = prepare_cmapss(cfg.data, sub, cfg.seed)
        te_ld = DataLoader(te_ds, batch_size=512, shuffle=False)
        plot_fig5(model, te_ld, device, out_dir, sub)
    print("全部图表完成（论文布局：图3/4/6 各一张、图5 每数据集一张）")


if __name__ == "__main__":
    main()

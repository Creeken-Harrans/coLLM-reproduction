"""最终结果汇总图：论文 vs 复现对比 + 消融效果 + 反思正确率。

输出（{out_dir}/figures/）:
- summary_vs_paper.png     RMSE/MAE 对比（论文 vs 复现，FD001/FD003 × A/B/C）
- summary_ablation.png     自反思消融（有/无 RMSE）+ 反思正确率
- summary_speedup.png      加速比对比（论文 vs 复现，对数坐标）
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from collm.config import get_config

# 输出根目录统一随配置 out_dir（结果 JSON 与图件路径一致）
OUT_DIR = Path(get_config("FD001").out_dir)
FIG_DIR = OUT_DIR / "figures"

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

# 统一配色（与 plot_results.py 一致：论文=蓝、复现=橙红、消融=绿/灰、反思=红）
C_PAPER = "#5b8db8"
C_OURS = "#e76f51"
C_WITH = "#2a9d8f"
C_WITHOUT = "#a8b6bd"
C_ACC = "#d62828"

# 论文数据（表 II 为准；FD003-C MAE 正文为 7.12 与表 II 7.04 不一致，
# 统一取表 II 7.04，论文内部矛盾记录于 PAPER_BENCHMARKS.md）
PAPER = {
    "FD001": {"A": (12.45, 9.13), "B": (12.40, 8.98), "C": (12.33, 8.86)},
    "FD003": {"A": (11.26, 7.42), "B": (11.11, 7.23), "C": (11.11, 7.04)},
}


def load(subset):
    return json.load(open(OUT_DIR / "results" / f"{subset}_results.json"))


def plot_vs_paper():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
    for ax, subset in zip(axes, ["FD001", "FD003"]):
        res = load(subset)
        combos = ["A", "B", "C"]
        x = np.arange(len(combos)); w = 0.35
        paper_rmse = [PAPER[subset][c][0] for c in combos]
        our_rmse = [res["combos"][c]["RMSE"] for c in combos]
        paper_mae = [PAPER[subset][c][1] for c in combos]
        our_mae = [res["combos"][c]["MAE"] for c in combos]
        b1 = ax.bar(x - w / 2, paper_rmse, w, label="论文", color=C_PAPER,
                    edgecolor="#333333", lw=0.5)
        b2 = ax.bar(x + w / 2, our_rmse, w, label="复现", color=C_OURS,
                    edgecolor="#333333", lw=0.5)
        for bars, vals in [(b1, paper_rmse), (b2, our_rmse)]:
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.12, f"{v:.2f}",
                        ha="center", fontsize=7.5)
        ax2 = ax.twinx()
        ax2.plot(x, paper_mae, "s--", color=C_WITH, ms=4, lw=1.2, label="论文 MAE")
        ax2.plot(x, our_mae, "o-", color="#e4572e", ms=4, lw=1.2, label="复现 MAE")
        ax2.set_ylabel("MAE"); ax2.set_ylim(5, 12)
        ax.set_xticks(x); ax.set_xticklabels([f"CoLLM-{c}" for c in combos])
        ax.set_ylabel("RMSE"); ax.set_title(f"{subset}（触发率 复现："
            f"{res['combos']['A']['lm_ratio']*100:.0f}/{res['combos']['B']['lm_ratio']*100:.0f}/"
            f"{res['combos']['C']['lm_ratio']*100:.0f}%）")
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    fig.suptitle("论文 vs 复现：CoLLM RMSE / MAE 对比（表 II）", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "summary_vs_paper.png", bbox_inches="tight")
    plt.close(fig)
    print(f"已保存 {FIG_DIR / 'summary_vs_paper.png'}")


def plot_ablation():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, subset in zip(axes, ["FD001", "FD003"]):
        res = load(subset)
        combos = ["A", "B", "C", "T3-09"]
        with_r = [res["combos"][c]["RMSE"] for c in combos]
        without = [res["combos"][c]["no_reflection_RMSE"] for c in combos]
        correct = [res["combos"][c]["reflect_correct"] * 100 for c in combos]
        x = np.arange(len(combos)); w = 0.35
        b1 = ax.bar(x - w / 2, with_r, w, label="有自反思", color=C_WITH,
                    edgecolor="#333333", lw=0.5)
        b2 = ax.bar(x + w / 2, without, w, label="无自反思", color=C_WITHOUT,
                    edgecolor="#333333", lw=0.5)
        ax.set_xticks(x); ax.set_xticklabels(combos)
        ax.set_ylabel("RMSE"); ax.set_title(f"{subset} 消融")
        for b, v in zip(list(b1) + list(b2), with_r + without):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15, f"{v:.2f}",
                    ha="center", fontsize=7.5)
        ax2 = ax.twinx()
        ax2.plot(x, correct, "D-", color=C_ACC, ms=4, lw=1.2, label="反思正确率(%)")
        if subset == "FD003":
            ax2.annotate("A 无触发（0 反思样本）", xy=(0, 0), xytext=(0.15, 12),
                         fontsize=7, color="#555555")
        ax2.axhline(70, ls="--", color="#999999", lw=0.8)
        ax2.text(3.35, 71, "论文标准 70%", fontsize=7, color="#555")
        ax2.set_ylabel("反思正确率 (%)"); ax2.set_ylim(0, 100)
        ax.legend(fontsize=8, loc="upper left")
        ax2.legend(fontsize=8, loc="upper right")
    fig.suptitle("自反思消融（论文：删除后 RMSE 变差 0.3-1.1%；复现 8.8-22.7%——"
                 "LM 弱于论文 → 融合收益更大）", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "summary_ablation.png", bbox_inches="tight")
    plt.close(fig)
    print(f"已保存 {FIG_DIR / 'summary_ablation.png'}")


def plot_speedup():
    """加速比对比（论文 vs 复现，对数坐标）+ 触发率。"""
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, subset in enumerate(["FD001", "FD003"]):
        res = load(subset)
        paper_sp = {"FD001": {"A": 3.88, "B": 2.08, "C": 1.26},
                    "FD003": {"A": 14.54, "B": 2.29, "C": 1.57}}[subset]
        x = np.arange(3) + i * 4.5
        our = [res["combos"][c]["speedup"] for c in ["A", "B", "C"]]
        pp = [paper_sp[c] for c in ["A", "B", "C"]]
        ax.bar(x - 0.3, pp, 0.6, label=f"{subset} 论文" if i == 0 else None,
               color=C_PAPER, edgecolor="#333333", lw=0.5)
        ax.bar(x + 0.3, our, 0.6, label=f"{subset} 复现" if i == 0 else None,
               color=C_OURS, edgecolor="#333333", lw=0.5)
        for xx, v in zip(x - 0.3, pp):
            ax.text(xx, v * 1.05, f"{v:.2f}×" if v < 100 else f"{v:.0f}×",
                    ha="center", fontsize=8)
        for xx, v in zip(x + 0.3, our):
            ax.text(xx, v * 1.05, f"{v:.2f}×" if v < 100 else f"{v:.0f}×",
                    ha="center", fontsize=8)
    ax.set_xticks([1, 5.5]); ax.set_xticklabels(["FD001", "FD003"])
    ax.set_ylabel("FLOPs 加速比（×，对数）"); ax.set_yscale("log")
    ax.set_title("加速比对比（FD001：复现触发率更高→加速更小；FD003：触发率更低→加速更大——"
                 "均为模型强度差的诚实后果）", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "summary_speedup.png", bbox_inches="tight")
    plt.close(fig)
    print(f"已保存 {FIG_DIR / 'summary_speedup.png'}")


if __name__ == "__main__":
    plot_vs_paper()
    plot_ablation()
    plot_speedup()
    print("汇总图全部完成")

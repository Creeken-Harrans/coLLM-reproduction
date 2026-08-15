"""评估口径分析：全窗口 vs 每发动机末端窗口（标准 C-MAPSS 协议）。

论文表 III 反思样本数（249/518 > 100 台）证明 CoLLM 组合为全窗口口径；
但表 II 基线列（One Fits All FT 12.34 等）是否为末端窗口口径值得核对——
本脚本对给定 LM/SM 权重同时报告两种口径，供文档记录。
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
from collm.models.small_model import SmallModel
from collm.models.large_model import LargeModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001", choices=["FD001", "FD003"])
    ap.add_argument("--model", default="large", choices=["small", "large"])
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    cfg = get_config(args.subset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr_ds, val_ds, te_ds, stats = prepare_cmapss(cfg.data, args.subset, cfg.seed)

    if args.model == "small":
        m = SmallModel(cfg.small, max_len=cfg.data.window).to(device)
    else:
        m = LargeModel(cfg.large, n_sensors=cfg.data.n_sensors).to(device)
    m.load_state_dict(torch.load(args.ckpt, map_location=device))
    m.eval()

    ld = DataLoader(te_ds, batch_size=512, shuffle=False)
    preds, trues, ends = [], [], []
    with torch.no_grad():
        for x, y in ld:
            x = x.to(device)
            out = m(x)
            if isinstance(out, tuple):
                out = out[0]
            preds.append(out.cpu().numpy())
            trues.append(y.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    # 每发动机取最后一个窗口（数据按 unit 顺序排列，unit 变化处即边界）
    # 用原始文件重建 unit 序列
    import numpy as np
    raw = np.loadtxt(Path(cfg.data.data_dir) / f"test_FD{args.subset[2:]}.txt")
    units = raw[:, 0].astype(np.int64)
    cyc = raw[:, 1].astype(np.int64)
    last_idx = []
    for i in range(len(units) - 1):
        if units[i] != units[i + 1]:
            last_idx.append(i)
    last_idx.append(len(units) - 1)
    # 窗口 i 的末端 cycle = 窗口起点 + 49；该 unit 最后一个窗口 = 末端 cycle == 该 unit 最大 cycle
    max_cycle_of = {}
    for i in last_idx:
        max_cycle_of[units[i]] = cyc[i]
    # 从 dataset 中取窗口：需要窗口的末端 cycle —— 由 data.py 逻辑重现
    wins_end = []
    cur = 0
    for i in range(len(units) - 1):
        if units[i] != units[i + 1]:
            n = i - cur + 1
            if n >= 50:
                for s in range(0, n - 50 + 1):
                    wins_end.append((units[i], cyc[s + 49]))
            cur = i + 1
    n = len(units) - cur
    if n >= 50:
        for s in range(0, n - 50 + 1):
            wins_end.append((units[-1], cyc[cur + s + 49]))
    wins_end = np.array(wins_end)
    is_last = np.array([c == max_cycle_of[u] for u, c in wins_end])

    def rep(name, mask):
        rmse = float(np.sqrt(np.mean((preds[mask] - trues[mask]) ** 2)))
        mae_ = float(np.mean(np.abs(preds[mask] - trues[mask])))
        print(f"{args.subset} {args.model} {name}: n={mask.sum()} RMSE {rmse:.3f} MAE {mae_:.3f}")

    rep("all-window", np.ones(len(preds), dtype=bool))
    rep("last-window-per-engine", is_last)


if __name__ == "__main__":
    main()

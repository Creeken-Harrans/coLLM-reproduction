"""CMAPSS 数据管线 — 严格按论文预处理。

论文数据流:
  原始序列 → 删除 7 个恒定传感器（1,5,6,10,16,18,19）→ 剩余 14 个传感器
  → z-score 标准化（x = (x-μ)/σ，μ/σ 在训练集上计算）
  → 滑动窗口（长度 50、步长 1）→ x ∈ R^{50×14}，标签为窗口末端 cycle 的 RUL。

标签约定（与文献 [26] DLformer 一致）:
  - 训练: 某 unit 的第 i 行 cycle 为 i（1-based），窗口 [i, i+49] 的 RUL
           = 该 unit 总寿命 − (i+49)。
  - 测试: RUL_FD00X 给出各测试 unit 末端 cycle 的 RUL，窗口向前递推。
"""
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from .config import DataConfig

# 21 个传感器中恒定、被移除的 7 个（论文 1-based 编号 → 文件列偏移 5）
KEEP_SENSOR_INDICES = [i for i in range(21) if (i + 1) not in (1, 5, 6, 10, 16, 18, 19)]


def _subset_num(subset: str) -> str:
    """'FD001' → '001'；'001' 原样返回。"""
    return subset[2:] if subset.upper().startswith("FD") else subset


def _load_raw(data_dir: str, split: str, subset: str) -> np.ndarray:
    """读取 CMAPSS txt。每行 26 列: unit, cycle, 3 操作设置, 21 传感器。"""
    return np.loadtxt(Path(data_dir) / f"{split}_FD{_subset_num(subset)}.txt")


def _extract_sensors(arr: np.ndarray) -> np.ndarray:
    """去掉 7 个恒定传感器，保留 14 维传感器数据。"""
    return arr[:, 5:][:, KEEP_SENSOR_INDICES]


def _unit_blocks(units: np.ndarray, cycles: np.ndarray):
    """把 (unit, cycle) 序列切成 (unit_id, cycles 数组, 行下标) 的块列表。"""
    blocks = []
    start = 0
    n = len(units)
    for i in range(1, n + 1):
        if i == n or units[i] != units[start]:
            blocks.append((units[start], cycles[start:i], slice(start, i)))
            start = i
    return blocks


def _build_windows(sens: np.ndarray, blocks, window: int, stride: int):
    """对每个 unit 滑窗。返回 (x: [N, window, 14], end_cycle: [N])。

    步长 1 → 每 unit 生成 max(0, n_cycles − window + 1) 个窗口。
    窗口末端 cycle = 窗口起点 cycle + window − 1。
    """
    xs, end_cycles = [], []
    for unit, cyc, slc in blocks:
        s = sens[slc]
        n = len(cyc)
        for i in range(0, n - window + 1, stride):
            xs.append(s[i:i + window])
            end_cycles.append(cyc[i + window - 1])
    if not xs:
        raise ValueError("窗口数为 0，请检查窗口长度与数据")
    return np.stack(xs), np.array(end_cycles), np.array([b[0] for b in blocks for _ in range(max(0, len(b[1]) - window + 1))])


def prepare_cmapss(cfg: DataConfig, subset: str, seed: int = 42):
    """完整预处理 → (train_ds, val_ds, test_ds, stats)。

    验证集：论文“随机选取 20%”，此处按 unit 划分（同一发动机的窗口不跨集）。
    """
    rng = np.random.default_rng(seed)

    # ---- 读取与去恒定传感器 ----
    tr = _load_raw(cfg.data_dir, "train", subset)
    te = _load_raw(cfg.data_dir, "test", subset)
    tr_sens = _extract_sensors(tr)
    te_sens = _extract_sensors(te)

    tr_units, tr_cycles = tr[:, 0].astype(np.int64), tr[:, 1].astype(np.int64)
    te_units, te_cycles = te[:, 0].astype(np.int64), te[:, 1].astype(np.int64)

    # ---- 训练/测试各自滑窗 ----
    tr_blocks = _unit_blocks(tr_units, tr_cycles)
    te_blocks = _unit_blocks(te_units, te_cycles)
    tr_x, tr_end_cyc, tr_w_units = _build_windows(tr_sens, tr_blocks, cfg.window, cfg.stride)
    te_x, te_end_cyc, _ = _build_windows(te_sens, te_blocks, cfg.window, cfg.stride)

    # ---- RUL 标签 ----
    # 训练：unit 总寿命 = 最大 cycle
    tr_life = {b[0]: int(b[1].max()) for b in tr_blocks}
    tr_y = np.array([tr_life[u] - c for u, c in zip(tr_w_units, tr_end_cyc)], dtype=np.float64)

    # 测试：RUL_FD00X 行序 = 测试文件中 unit 的首次出现序
    te_final_rul = np.loadtxt(Path(cfg.data_dir) / f"RUL_FD{_subset_num(subset)}.txt")
    te_unit_order = {}
    for u in te_units:
        if u not in te_unit_order:
            te_unit_order[u] = len(te_unit_order)
    te_cycles_per_unit = {b[0]: len(b[1]) for b in te_blocks}
    te_y = np.array([
        te_final_rul[te_unit_order[u]] + (te_cycles_per_unit[u] - c)
        for u, c in zip(_w_units(te_blocks, cfg.window, cfg.stride), te_end_cyc)
    ], dtype=np.float64)

    # ---- 20% 验证集（先划分，后标准化——统计量只用 train 部分，避免泄漏）----
    # 论文文字为“随机选取 20%”，但 stride=1 的重叠窗口下按样本随机会产生
    # 严重数据泄漏（相邻窗口共享 49/50 数据，val 与 train 几乎相同，早停失效，
    # 实测 val RMSE 2.08 vs 按 unit 的 14.6）。因此按 unit 随机 20% 划分（防泄漏）。
    all_units = np.unique(tr_units)
    n_val = int(round(len(all_units) * cfg.val_ratio))
    val_units = set(rng.choice(all_units, size=n_val, replace=False).tolist())
    val_mask = np.array([u in val_units for u in tr_w_units])
    tr_mask = ~val_mask

    # ---- 标准化 ----
    # 论文公式 15（英文原文）："mean and standard deviation computed over the
    # entire dataset"。训练场景下"整个数据集"= 全部训练数据（划分 val 之前，
    # 不含 test——论文语境为最小化 train/test 分布不匹配）。
    # 'all_train'=train+val 全部 | 'train'=仅 val 划分外 | 'entire'=含 test（实测更差）。
    # 注：μ/σ 在滑窗后的窗口数组上计算（原始行被 1-50 个窗口重复计入，中段权重
    # 约 50× 边缘）；对线性漂移传感器加权均值与原始行均值精确相等，方差仅 ~1%
    # 量级偏移（子代理推导），对 RMSE 影响可忽略——与论文 "entire dataset" 字面一致。
    if cfg.norm_mode == "all_train":
        mu = tr_x.reshape(-1, tr_x.shape[-1]).mean(axis=0)
        sigma = tr_x.reshape(-1, tr_x.shape[-1]).std(axis=0)
    elif cfg.norm_mode == "entire":
        mu = np.concatenate([tr_x, te_x]).reshape(-1, tr_x.shape[-1]).mean(axis=0)
        sigma = np.concatenate([tr_x, te_x]).reshape(-1, tr_x.shape[-1]).std(axis=0)
    else:
        mu = tr_x[tr_mask].reshape(-1, tr_x.shape[-1]).mean(axis=0)
        sigma = tr_x[tr_mask].reshape(-1, tr_x.shape[-1]).std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    tr_x = (tr_x - mu) / sigma
    te_x = (te_x - mu) / sigma

    # ---- RUL 截断（cap=125 分段线性：像素级证实 + 文献 [26] 惯例，REVISIONS #45.4）----
    if cfg.rul_cap is not None:
        tr_y = np.minimum(tr_y, cfg.rul_cap)
        te_y = np.minimum(te_y, cfg.rul_cap)

    tr_ds = CmapssDataset(tr_x[tr_mask], tr_y[tr_mask])
    val_ds = CmapssDataset(tr_x[val_mask], tr_y[val_mask])
    te_ds = CmapssDataset(te_x, te_y)

    stats = {
        "subset": subset,
        "train_windows": int(tr_mask.sum()), "val_windows": int(val_mask.sum()),
        "test_windows": len(te_ds),
        "train_units": int(len(all_units) - n_val), "val_units": int(n_val),
        "test_units": len(te_unit_order),
        "test_engines": len(te_blocks),
        # 标准化统计量（当前 norm_mode 下）——供可视化等下游复用，保证与训练一致
        "mu": mu, "sigma": sigma,
    }
    # 预处理数据落盘（data/processed，供完整流程复用/审计）
    try:
        out_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / f"{subset}_processed.npz",
            x_train=tr_x[tr_mask], y_train=tr_y[tr_mask],
            x_val=tr_x[val_mask], y_val=tr_y[val_mask],
            x_test=te_x, y_test=te_y,
            train_unit_ids=tr_w_units[tr_mask], val_unit_ids=tr_w_units[val_mask],
            mu=mu, sigma=sigma,
        )
        (out_dir / f"{subset}_meta.json").write_text(
            json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in stats.items() if k not in ("mu", "sigma")},
                       indent=1, ensure_ascii=False))
    except Exception as e:  # 落盘失败不影响训练
        print(f"[data] 预处理落盘失败: {e}")
    return tr_ds, val_ds, te_ds, stats


def _w_units(blocks, window: int, stride: int):
    """与 _build_windows 对应的每窗口 unit 序列。"""
    out = []
    for unit, cyc, _ in blocks:
        n = len(cyc)
        out.extend([unit] * (max(0, n - window + 1) // stride))
    return np.array(out)


class CmapssDataset(Dataset):
    """滑窗后的 CMAPSS 数据集。x ∈ R^{window×14}，y = RUL 标量。"""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

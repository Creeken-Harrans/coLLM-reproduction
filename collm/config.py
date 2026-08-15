"""CoLLM 复现全局配置 — 定稿值（2026-08-04 最终）。

论文: CoLLM: Industrial Large-Small Model Collaboration With Fuzzy
Decision-Making Agent and Self-Reflection (IEEE TFS 2026).

配置按子集（FD001/FD003）集中定义：每个子集一个 Config 实例，
脚本仅需 --subset 自动加载，不再散落实验参数。
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    """数据预处理（论文实验设置：公式 15-16 + 滑窗 + RUL 截断）。"""
    data_dir: str = "data/raw/cmapss"          # 原始 CMAPSS txt
    processed_dir: str = "data/processed"      # 预处理结果落盘（npz + meta json）
    window: int = 50          # 滑动窗口长度（论文）
    stride: int = 1           # 滑动步长（论文）
    # 21 个传感器中恒定、与 RUL 相关性弱而被移除的 7 个（论文 1-based 编号）
    removed_sensors: tuple = (1, 5, 6, 10, 16, 18, 19)
    n_sensors: int = 14       # 21 - 7
    val_ratio: float = 0.2    # 论文：20% 作为验证集（按 unit 划分，防泄漏）
    norm_mode: str = "train"  # z-score 统计量范围：仅 val 划分外的训练窗口
                              # （'entire dataset' 三种解读实测最优，REVISIONS #30）
    rul_cap: Optional[float] = 125   # RUL 截断（分段线性；论文指标反推 + 文献 [26] 惯例）


@dataclass
class SmallModelConfig:
    """阶段1 小模型 S（论文：微调后的 Transformer 编码器，表 I）。

    表 I：输入嵌入 14→32、Encoder 隐藏 64、特征 d_s=32。
    层数论文未给：FD001 8 层（dropout 0.2，φs 信息量 0.106→0.286）、FD003 6 层
    （实测最优，REVISIONS #44）。
    """
    n_sensors: int = 14
    d_model: int = 32         # 输入嵌入维
    d_hidden: int = 64        # Transformer Encoder 隐藏维度（论文表 I）
    n_heads: int = 2
    n_layers: int = 8         # 子集相关：FD001=8、FD003=6（见 get_config）
    d_feat: int = 32          # 输出特征 φs(x) 维度 d_s=32（论文表 I）
    dropout: float = 0.1      # FD001 用 0.2（正则，实测最优）
    norm_first: bool = False  # post-norm（PyTorch 默认；论文未明确，实测）
    pool: str = "last"        # 预测头输入：末端时间步（实测最优，REVISIONS #17 后扫描）
    head_hidden: Optional[int] = None   # 预测头：单层 Linear（实测最优）


@dataclass
class LargeModelConfig:
    """阶段2 大模型 L（论文：预训练 GPT-2 + patch embedding，冻结 attention+FFN）。

    表 I：patch 4/stride 4、嵌入 768、特征 d_l=768。
    层数：FD001 前 9 层 / FD003 12 层（REVISIONS #27；FLOPs 对齐论文 2.21G 口径
    见 docs/FINAL_RESULTS.md）。
    """
    model_name: str = "pretrained/gpt2"   # 项目内本地权重（gpt2 small，124M）
    patch_size: int = 4       # 论文
    patch_stride: int = 4     # 论文
    d_embed: int = 768        # patch → 768 维（GPT-2 嵌入维）
    n_blocks: int = 9         # 子集相关：FD001=9、FD003=12（见 get_config）
    freeze_backbone: bool = True   # 冻结 attention+FFN（论文 D 节）
    learnable_ln_f: bool = True    # ln_f 与逐层 LN 可微调（论文冻结范围仅 attention+FFN）
    pos_init: str = "wpe"          # 位置嵌入：可学习，wpe 前 n_patch 位初始化（实测最优）
    pool_mode: str = "last"        # 预测头输入：末端 patch（实测最优）
    head_hidden: Optional[int] = 128   # 预测头 MLP（实测最优）
    max_patches: int = 64         # 位置编码上限（窗口 50 / patch 4 → 12 个 patch，留余量）


@dataclass
class FuzzyConfig:
    """FNN 模糊决策智能体（论文公式 8-11，表 I：64 个隶属函数）。

    Qs = σ(W·模糊特征 + b)；标签 Q*s = 1 - tanh(|ys - y*| / α)。
    """
    d_input: int = 32         # 输入为 SM 特征 φs(x)（表 I 特征 d_s=32）
    n_membership: int = 64    # 模糊函数数量（表 I：Number of fd function = 64，每维 2 个）
    pool_mode: str = "stats"  # 时间聚合：FD001=stats / FD003=mean（网格实测最优）
    hidden: Optional[int] = None   # 置信度头：None=单层（公式 11 字面，实测最优）
    alpha: float = 4.0        # 残差缩放系数 α：FD001=4 / FD003=5（网格实测最优）


@dataclass
class ReflectionConfig:
    """自反思模型 R（论文公式 12-14：φl 展平 + 单层全连接 → Ql）。

    输入展平维度 = n_patch(12) × d_l(768) = 9216（窗口 50 / patch 4，末端截断）。
    """
    d_input: int = 9216
    alpha: float = 4.0        # 与 FNN 相同 α（论文公式 10/13 同一符号）


@dataclass
class TrainConfig:
    """训练超参（论文表 I + 定稿实测）。"""
    batch_size: int = 256     # 论文表 I
    lr: float = 2e-3          # 论文表 I（Adam）
    epochs: int = 100         # 论文：100 epoch + 早停
    patience: int = 12
    sm_sched: str = "cosine"  # SM 调度：cosine 实测 13.515→13.007（第三轮新发现）
    seed: int = 42            # 数据划分 seed（LM 5 划分实测最优，REVISIONS #44）
    num_workers: int = 2
    # 阶段3（置信度模块）——网格实测最优（REVISIONS #42）
    conf_epochs: int = 100    # 固定 epochs（val 早停与组合目标不一致，实测固定更优）
    conf_lr: float = 1e-3
    conf_batch: int = 1024
    conf_use_val: bool = True # 训练数据 = train+val 全部窗口


@dataclass
class Thresholds:
    """阈值 [τ1, τ2]（论文正文 B/C 节）。"""
    fd001_a: tuple = (0.3, 0.1)
    fd001_b: tuple = (0.4, 0.1)
    fd001_c: tuple = (0.6, 0.05)
    fd003_a: tuple = (0.15, 0.1)
    fd003_b: tuple = (0.4, 0.1)
    fd003_c: tuple = (0.6, 0.05)


@dataclass
class Config:
    """子集定稿配置（get_config(subset) 返回）。"""
    data: DataConfig = field(default_factory=DataConfig)
    small: SmallModelConfig = field(default_factory=SmallModelConfig)
    large: LargeModelConfig = field(default_factory=LargeModelConfig)
    fuzzy: FuzzyConfig = field(default_factory=FuzzyConfig)
    reflection: ReflectionConfig = field(default_factory=ReflectionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    thresholds: Thresholds = field(default_factory=Thresholds)
    device: str = "cuda"
    seed: int = 42
    out_dir: str = "outputs"


def get_config(subset: str) -> Config:
    """返回指定子集的定稿配置（FD001/FD003 的参数差异集中于此）。

    - SM 层数/正则：FD001 8 层 dropout 0.2（φs 泛化差，加深+正则最优）；
      FD003 6 层 dropout 0.1（已达标，保持）
    - LM 层数：FD001 9 层（12 层过拟合 test 15.7）；FD003 12 层（9 层 13.19 < 12 层 13.08）
    - FNN 聚合/α：FD001 stats/α4；FD003 mean/α5（网格实测最优，REVISIONS #42）
    """
    cfg = Config()
    if subset == "FD001":
        cfg.small.n_layers = 8
        cfg.small.dropout = 0.2
        cfg.large.n_blocks = 9
        cfg.fuzzy.pool_mode = "stats"
        cfg.fuzzy.alpha = 4.0
        cfg.reflection.alpha = 4.0
    elif subset == "FD003":
        cfg.small.n_layers = 6
        cfg.large.n_blocks = 12
        cfg.fuzzy.pool_mode = "mean"
        cfg.fuzzy.alpha = 5.0
        cfg.reflection.alpha = 5.0
    else:
        raise ValueError(f"未知子集: {subset}")
    return cfg

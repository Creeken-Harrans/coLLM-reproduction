"""CoLLM 复现全局配置 — 定稿值（2026-08-29 第四轮）。

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
    norm_mode: str = "train"  # z-score 统计量范围：仅 val 划分外的训练窗口（防泄漏定稿）
                              # 可选值 'all_train'/'entire' 含 val/test 统计量，属泄漏，
                              # 仅历史对照（REVISIONS #30 实测 'train' 最优），勿启用
    rul_cap: Optional[float] = 125   # RUL 截断（分段线性；论文指标反推 + 文献 [26] 惯例）

    # ---- 短测试序列（周期数 < window）的处理口径 ----
    # 论文只规定"窗口 50 / 步长 1"，未交代不足 50 周期的测试序列怎么办。既有实现
    # 由 max(0, n - window + 1) 静默丢弃：FD001 丢 7 台、FD003 丢 3 台，且被丢的恰是
    # 健康度最高的发动机（末端 RUL 均值 119/136 vs 保留的 72/73）——测试集因此系统性
    # 偏向不健康发动机（corr(测试序列长度, 末端 RUL) = -0.60，REVISIONS #53）。
    #   'drop'     = 丢弃（论文口径下的既有行为；默认值，保证既有结果逐位不变）
    #   'minpad'   = 前置填充至恰好 window → 每台 1 个末端窗口
    #   'percycle' = 每个 cycle 一个窗口，按缺失量前置填充（早期 cycle 填充量极大）
    # 注意：'drop' 之外的口径需要向模型喂训练时从未见过的填充序列（OOD），实测结果
    # 由填充方式主导而非模型能力（REVISIONS #53），仅供口径审计，勿作为论文对照。
    short_seq: str = "drop"
    short_pad: str = "replicate"   # 'zero'（z-score 后 0 = 训练均值）| 'replicate'（重复首行）


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
    """阶段2 大模型 L（论文：One Fits All = GPT4TS，冻结 attention+FFN）。

    表 I：patch 4/stride 4、嵌入 768、特征 d_l=768。
    输入编码为 One Fits All 官方结构（Conv1d TokenEmbedding + 固定正弦位置嵌入 +
    ReplicationPad 13 patch），FD001/FD003 均 12 层（REVISIONS #50）。
    """
    model_name: str = "pretrained/gpt2"   # 项目内本地权重（gpt2 small，124M）
    patch_size: int = 4       # 论文
    patch_stride: int = 4     # 论文
    d_embed: int = 768        # patch → 768 维（GPT-2 嵌入维）
    n_blocks: int = 12        # 子集相关：FD001=12、FD003=12（见 get_config）
    head_hidden: Optional[int] = 128   # 预测头 MLP（实测最优）


@dataclass
class FuzzyConfig:
    """FNN 模糊决策智能体（论文公式 8-11，表 I：64 个隶属函数）。

    Qs = σ(W·模糊特征 + b)；标签 Q*s = 1 - tanh(|ys - y*| / α)。
    """
    d_input: int = 32         # 输入为 SM 特征 φs(x)（表 I 特征 d_s=32）
    n_membership: int = 64    # 模糊函数数量（表 I：Number of fd function = 64，每维 2 个）
    pool_mode: str = "stats"  # 时间聚合：FD001=stats / FD003=mean（网格实测最优）
    hidden: Optional[int] = None   # 置信度头：None=单层（公式 11 字面，实测最优）
    alpha: float = 6.0        # 残差缩放系数 α：FD001=6 / FD003=6（第三轮扫描，见 get_config）


@dataclass
class ReflectionConfig:
    """自反思模型 R（论文公式 12-14：φl 展平 + 单层全连接 → Ql）。

    输入展平维度 = n_patch(13) × d_l(768) = 9984（窗口 50 / patch 4，ReplicationPad 尾补）。
    """
    d_input: int = 9984
    alpha: float = 6.0        # 与 FNN 相同 α（论文公式 10/13 同一符号）


@dataclass
class TrainConfig:
    """训练超参（论文表 I + 定稿实测）。"""
    batch_size: int = 256     # 论文表 I
    lr: float = 2e-3          # 论文表 I（Adam）
    epochs: int = 100         # 论文：100 epoch + 早停
    patience: int = 12
    sm_sched: str = "cosine"  # SM 调度：cosine 实测 13.515→13.007（第三轮新发现）
    seed: int = 42            # 数据划分 seed（统一 42；LM 单 seed 协议 REVISIONS #48）
    num_workers: int = 0      # 0=确定性（torch 2.9 num_workers>0 下 shuffle 非确定，见 REVISIONS #51）
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
    - LM 层数：FD001/FD003 均 12 层（One Fits All 官方 GPT4TS 结构，REVISIONS #50；
      13 patch 口径对应论文 2.21G FLOPs）
    - FNN 聚合/α：FD001 stats/α6；FD003 mean/α6（第三轮 α×pool 扫描 val 最优，
      EXPERIMENTS_ROUND3.md；扫描与 train_conf 配方逐位一致后复选，
      REVISIONS #48——Ql 饱和悬崖对训练配方敏感）
    - LM seed 协议：统一单 seed 42（LM 的 val-best 已证不可靠——val 幸运陷阱
      REVISIONS #29/#46.2/#48；SM 才用多 seed val-best）
    """
    cfg = Config()
    if subset == "FD001":
        cfg.small.n_layers = 8
        cfg.small.dropout = 0.2
        cfg.large.n_blocks = 12
        cfg.fuzzy.pool_mode = "stats"
        cfg.fuzzy.alpha = 6.0
        cfg.reflection.alpha = 6.0
    elif subset == "FD003":
        cfg.small.n_layers = 6
        cfg.small.dropout = 0.1
        cfg.large.n_blocks = 12
        cfg.fuzzy.pool_mode = "mean"
        cfg.fuzzy.alpha = 6.0
        cfg.reflection.alpha = 6.0
    else:
        raise ValueError(f"未知子集: {subset}")
    return cfg

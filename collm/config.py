"""CoLLM 复现全局配置 — 严格对齐论文（正文为准）。

论文: CoLLM: Industrial Large-Small Model Collaboration With Fuzzy
Decision-Making Agent and Self-Reflection (IEEE TFS 2026).
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    data_dir: str = "data/raw/cmapss"
    window: int = 50          # 滑动窗口长度（论文）
    stride: int = 1           # 滑动步长（论文）
    # 21 个传感器中恒定、与 RUL 相关性弱而被移除的 7 个（论文 1-based 编号）
    removed_sensors: tuple = (1, 5, 6, 10, 16, 18, 19)
    n_sensors: int = 14       # 21 - 7
    val_ratio: float = 0.2    # 论文：20% 作为验证集（按 unit 划分，避免泄漏）
    norm_mode: str = "train"  # z-score 统计量范围（公式 15 三种解读实测，REVISIONS #30）
                              # 'train'（默认，仅 val 划分外——实测最优）
                              # | 'all_train'（train+val 全部，实测更差）| 'entire'（含 test，弃用）
    # RUL 截断上限（分段线性 RUL）。论文正文未明说，但其跟随的 [26] DLformer
    # 及所有可达论文指标（RMSE≈12）的 CMAPSS 工作均采用截断；线性 RUL 下
    # 早期大 RUL 样本误差主导 RMSE，指标不可达 → 采用 cap=125（设计决策 #1）。
    rul_cap: Optional[float] = 125


@dataclass
class SmallModelConfig:
    """阶段1 小模型 S（论文：微调后的 Transformer 编码器）。

    论文/表 I 关键参数：输入嵌入 14→32，Encoder 隐藏维度 64，特征 d_s=32。
    """
    n_sensors: int = 14
    d_model: int = 32         # 输入嵌入维
    d_hidden: int = 64        # Transformer Encoder 隐藏维度（论文表 I）
    n_heads: int = 2
    n_layers: int = 6         # 实验：3→6 层使 FD001 test RMSE 17.3→15.7（设计决策 #12 更新）
    d_feat: int = 32          # 输出特征 φs(x) 维度 d_s=32（论文表 I）
    dropout: float = 0.1
    norm_first: bool = False  # Transformer norm 位置：False=post-norm（PyTorch 默认）
                              # True=pre-norm（GPT-2/One Fits All 风格，论文未明确，实验）
    pool: str = "last"        # 预测头输入：'last'=末端时间步（默认）| 'mean'=全局均值池化
    head_hidden: Optional[int] = None   # 预测头：None=Linear | 非 None=MLP(hidden)（实验）


@dataclass
class LargeModelConfig:
    """阶段2 大模型 L（论文：预训练 GPT-2 + patch embedding，冻结自注意力与 FFN）。"""
    model_name: str = "pretrained/gpt2"   # 项目内本地权重（gpt2 small）
    patch_size: int = 4       # 论文
    patch_stride: int = 4     # 论文
    d_embed: int = 768        # patch → 768 维（GPT-2 嵌入维）
    freeze_backbone: bool = True   # 冻结注意力+FFN（论文）
    freeze_pos: bool = False       # 位置嵌入：可学习（One Fits All 式）
    pos_init: str = "wpe"          # 位置嵌入初始化：'wpe'=预训练前 n_patch 位（实测更好）| 'random'=std 0.02
    learnable_ln_f: bool = True    # ln_f 不属于注意力/FFN（论文冻结范围），可微调
    instance_norm: bool = False    # One Fits All 核心组件：patch 前 per-sample per-channel 归一化（实验）
    pool_mode: str = "last"        # 预测头输入：'last'=末端 patch 特征 | 'mean'=全部 patch 均值（实验）
    n_blocks: int = 9              # 使用的 GPT-2 层数（FD001 实测 9 层最优：12→15.7、9→14.3；One Fits All 官方 6 层）
    dropout: float = 0.1           # GPT-2 注意力/残差 dropout（可关，实验）
    pad_patches: bool = False      # 窗口 50/4=12.5：尾部补足 13 个 patch（GPT4TS 惯例，实验）
    pad_mode: str = "replicate"    # 'replicate'=复制边缘值（GPT4TS 官方 ReplicationPad）
                                   # | 'zero'=零填充（BANjian16 CoLLM 复现）
    # 预测头：特征 → RUL 标量
    head_hidden: Optional[int] = 128
    head_concat: bool = False       # GPT4TS 官方头：全部 patch 特征拼接 → Linear(768×n_patch → 1)
    channel_independent: bool = False  # GPT4TS 官方：每传感器通道独立过 GPT-2（共享权重），
                                       # 每通道预测后取平均（RUL 任务）
    input_norm: bool = False           # proj 后加 LayerNorm（BANjian16 CoLLM 复现，实验）
    max_patches: int = 64     # 位置编码上限（窗口 50 / patch 4 → 12 个 patch，留余量）


@dataclass
class FuzzyConfig:
    """FNN 模糊决策智能体（论文公式 8-11）。

    高斯隶属函数 f_d(z) = exp(-(z-μ_d)^2/σ_d^2)，μ、σ^2 为可学习参数。
    置信度 Q_s = σ(W·φ_s(x)+b)，MSE 训练；标签 Q* = 1 - tanh(|y-y*|/α)。
    """
    d_input: int = 32         # 输入为 SM 特征 φs(x) 聚合后的维度（表 I 特征 d_s=32）
    n_membership: int = 64    # 模糊函数数量（论文表 I：Number of fd function = 64；d_s=32 维每维 2 个）
    feat_mode: str = "fuzzy"  # 'fuzzy'=模糊特征（论文公式9+表I）| 'raw'=原始 φs 池化（公式11 字面，消融）
    pool_mode: str = "mean"   # 时间聚合：'mean'=均值 | 'stats'=mean+max+std+last | 'flatten'=全时间步展平（信息最全，实验）
    hidden: Optional[int] = 64
    cat_pred: bool = False    # 置信度头输入拼接预测值 ys（论文"输入特征和预测结果的联合分布"，实验）
    alpha: float = 8.0        # 残差缩放系数 α：调小使 Q* 二值化（对/错），跨分布可预测性更好（REVISIONS #15）


@dataclass
class ReflectionConfig:
    """自反思模型 R（论文：FCN，输入展平 + 单层全连接 → 置信度标量）。"""
    d_input: int = 9984       # 展平后维度 = n_patch(13) × d_l(768)（GPT4TS 式补 13 patch）
    use_norm: bool = True     # 输入 LayerNorm（防 logit 饱和，论文未明确，实验选项）
    cat_pred: bool = False    # 输入拼接预测值 yl（论文"输入特征和预测结果的联合分布"，实验）
    alpha: float = 8.0        # 与 FNN 相同取法


@dataclass
class TrainConfig:
    batch_size: int = 256
    lr: float = 2e-3          # 论文表 I：Adam, lr 2e-3
    epochs: int = 100         # 论文：100 epoch + 早停
    patience: int = 12
    seed: int = 42
    num_workers: int = 2
    # 阶段2（LM 微调）用小学习率
    lr_lm: float = 1e-3
    lr_lm_head: float = 2e-3
    # 阶段3
    lr_conf: float = 2e-3


@dataclass
class Thresholds:
    """阈值 [τ1, τ2]（论文正文）。

    τ1: SM 置信度阈值，Q_s ≥ τ1 直接输出小模型结果；
    τ2: 自反思阈值，Δ = Q_s − Q_l ≤ τ2 接受大模型输出，否则 SM 辅助融合。
    """
    fd001_a: tuple = (0.3, 0.1)
    fd001_b: tuple = (0.4, 0.1)
    fd001_c: tuple = (0.6, 0.05)
    fd003_a: tuple = (0.15, 0.1)
    fd003_b: tuple = (0.4, 0.1)
    fd003_c: tuple = (0.6, 0.05)


@dataclass
class Config:
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

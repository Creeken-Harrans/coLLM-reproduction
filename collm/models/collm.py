"""CoLLM 框架组装（论文图 1 / Algorithm 1 的推理路由见 scripts/assess/evaluate.py::combine）。

推理规则（论文明确）:
    y_final = ys                    若 Q_s ≥ τ1        （小模型直接退出）
            = yl                    若 Q_s < τ1 且 Δ ≤ τ2（接受大模型）
            = (ys + yl)/2           若 Q_s < τ1 且 Δ > τ2（SM 辅助融合）
其中 Δ = Q_s − Q_l。

三阶段训练（论文）:
    阶段1: 只训 SM（scripts/train/train_small.py，直接操作 self.small）
    阶段2: 只训 LM 可训练部分（scripts/train/train_large.py，直接操作 self.large）
    阶段3: 只训 FNN 与自反思（scripts/train/train_conf.py，直接操作 self.fuzzy/self.reflection）
"""
import torch.nn as nn

from ..config import Config
from .small_model import SmallModel
from .large_model import LargeModel
from .fuzzy import FuzzyAgent
from .reflection import ReflectionModel


class CoLLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.small = SmallModel(cfg.small, max_len=cfg.data.window)
        self.large = LargeModel(cfg.large, n_sensors=cfg.data.n_sensors)
        self.fuzzy = FuzzyAgent(cfg.fuzzy)
        self.reflection = ReflectionModel(cfg.reflection)

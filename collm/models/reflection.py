"""自反思模型 R（论文：FCN）。

论文要点:
  - 输入为大模型潜在特征 φl(x) ∈ R^{t×d_l}（d_l=768）；
  - 沿时间维度展平为静态一维特征向量，经单层全连接投影为置信度标量 Q_l；
  - 监督信号基于大模型预测误差：Q*_l = 1 - tanh(|y_l - y*| / α)；
  - 训练目标 MSE: L = (1/N) Σ (Q_li - Q*_li)²。
"""
import torch
import torch.nn as nn

from ..config import ReflectionConfig


class ReflectionModel(nn.Module):
    def __init__(self, cfg: ReflectionConfig):
        super().__init__()
        self.cfg = cfg
        # 输入标准化：φl 为 GPT-2 高维特征，展平后范数大导致 logit 饱和
        # （实测 Ql 恒 1.0，见 REVISIONS #14）。LayerNorm 不改论文结构（展平+单层全连接）。
        self.norm = nn.LayerNorm(cfg.d_input) if cfg.use_norm else nn.Identity()
        # 单层全连接投影（论文公式 12）：展平 t×d_l → 标量
        self.proj = nn.Linear(cfg.d_input, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: φl(x) (B, t, d_l) → Q_l: (B,)"""
        b = feat.shape[0]
        f = feat.reshape(b, -1)                    # 展平（论文）
        f = self.norm(f)                           # 防饱和（可选）
        return torch.sigmoid(self.proj(f).squeeze(-1))

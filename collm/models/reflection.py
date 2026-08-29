"""自反思模型 R（论文公式 12-14：φl 展平 + 单层全连接 → Ql）。

论文要点:
  - 输入为大模型潜在特征 φl(x) ∈ R^{t×d_l}（d_l=768，t=12 patch）；
  - 沿时间维度展平为静态一维特征向量（9216 维），经单层全连接投影为
    置信度标量 Q_l（公式 12）；
  - 监督信号基于大模型预测误差：Q*_l = 1 - tanh(|y_l - y*| / α)（公式 13）；
  - 训练目标 MSE: L = (1/N) Σ (Q_li - Q*_li)²（公式 14）。
"""
import torch
import torch.nn as nn

from ..config import ReflectionConfig


class ReflectionModel(nn.Module):
    def __init__(self, cfg: ReflectionConfig, use_ln: bool = False):
        super().__init__()
        self.cfg = cfg
        self.use_ln = use_ln
        # 单层全连接投影（论文公式 12）：展平 t×d_l → 标量
        # 可选 LayerNorm：φl 高维特征范数大 → 无 LN 时 sigmoid 饱和（REVISIONS #46/#50）
        if use_ln:
            self.ln = nn.LayerNorm(cfg.d_input)
        self.proj = nn.Linear(cfg.d_input, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: φl(x) (B, t, d_l) → Q_l: (B,)"""
        b = feat.shape[0]
        f = feat.reshape(b, -1)                    # 展平（论文）
        if self.use_ln:
            f = self.ln(f)
        return torch.sigmoid(self.proj(f).squeeze(-1))

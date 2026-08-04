"""小模型 S（论文：微调后的 Transformer 编码器）。

架构（对齐论文表 I 与正文）:
  x ∈ R^{50×14} ──线性嵌入 14→32──▶ 每时间步 32 维
    ──▶ TransformerEncoder(d_model=32, nhead=2, dim_ff=64, N 层, post-norm)
    ──▶ φs(x) ∈ R^{50×32}            （特征 d_s=32，表 I）
    ──▶ 预测头（末端时间步特征，单层 Linear）──▶ ys ∈ R

论文阶段1：端到端最小化预测误差，只更新 S 与预测头。
层数：FD001=8（dropout 0.2）、FD003=6（get_config 定稿，REVISIONS #44）。
"""
import torch
import torch.nn as nn
from ..config import SmallModelConfig


class SmallModel(nn.Module):
    def __init__(self, cfg: SmallModelConfig, max_len: int = 50):
        super().__init__()
        self.cfg = cfg
        # 每时间步嵌入：14 → 32
        self.input_proj = nn.Linear(cfg.n_sensors, cfg.d_model)
        # 位置编码（可学习，std 0.02 初始化）：标准 Transformer 架构惯例
        # （无位置编码时置换等变、时序顺序信息丢失——REVISIONS #16）
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, cfg.d_model))
        nn.init.normal_(self.pos_embed, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_hidden,   # 隐藏维度 64（论文表 I）
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=cfg.norm_first,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        # 输出特征 φs(x)（编码器输出已为 32 维，与 d_feat 一致时恒等）
        if cfg.d_model != cfg.d_feat:
            self.feat_proj = nn.Linear(cfg.d_model, cfg.d_feat)
        else:
            self.feat_proj = nn.Identity()
        # 预测头：末端时间步特征 → RUL（单层 Linear，实测最优）
        self.predictor = nn.Linear(cfg.d_feat, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (ys, φs(x))。x: (B, T, 14)"""
        h = self.input_proj(x)                 # (B, T, 32)
        h = h + self.pos_embed[:, : h.shape[1]]
        h = self.encoder(h)                    # (B, T, 32)
        feat = self.feat_proj(h)               # (B, T, d_s)
        z = feat[:, -1]                        # 末端时间步（实测最优）
        ys = self.predictor(z)                 # (B, 1)
        return ys.squeeze(-1), feat            # (B,), (B, T, d_s)

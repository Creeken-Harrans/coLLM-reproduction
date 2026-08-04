"""小模型 S（论文：微调后的 Transformer 编码器）。

架构（对齐论文表 I 与正文）:
  x ∈ R^{50×14} ──线性嵌入 14→32──▶ 每时间步 32 维
    ──▶ TransformerEncoder(d_model=32, nhead=2, dim_ff=64, 3 层)
    ──▶ φs(x) ∈ R^{50×32}            （特征 d_s=32）
    ──▶ 预测头（末端时间步特征）──▶ ys ∈ R

论文阶段1：端到端最小化预测误差，只更新 S 与预测头。
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
        # 位置编码（可学习）：标准 Transformer 要求（论文 "Transformer encoder"
        # 惯例包含位置编码；无位置编码时置换等变、时序顺序信息丢失——REVISIONS #16）
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, cfg.d_model))
        nn.init.normal_(self.pos_embed, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_hidden,   # 隐藏维度 64（论文表 I）
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=cfg.norm_first,      # pre-norm（GPT-2 风格）实验
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        # 输出特征投影到 d_feat=32（编码器输出已为 32，此处保持接口一致）
        if cfg.d_model != cfg.d_feat:
            self.feat_proj = nn.Linear(cfg.d_model, cfg.d_feat)
        else:
            self.feat_proj = nn.Identity()
        # 预测头：池化后特征 → RUL 标量（head_hidden=None=Linear；否则 MLP）
        if cfg.head_hidden:
            self.predictor = nn.Sequential(
                nn.Linear(cfg.d_feat, cfg.head_hidden), nn.GELU(),
                nn.Linear(cfg.head_hidden, 1))
        else:
            self.predictor = nn.Linear(cfg.d_feat, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (ys, φs(x))。x: (B, T, 14)"""
        h = self.input_proj(x)                 # (B, T, 32)
        h = h + self.pos_embed[:, : h.shape[1]]
        h = self.encoder(h)                    # (B, T, 32)
        feat = self.feat_proj(h)               # (B, T, d_s)
        if self.cfg.pool == "mean":
            z = feat.mean(dim=1)               # 全局均值池化
        else:
            z = feat[:, -1]                    # 末端时间步（默认）
        ys = self.predictor(z)                 # (B, 1)
        return ys.squeeze(-1), feat            # (B,), (B, T, d_s)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

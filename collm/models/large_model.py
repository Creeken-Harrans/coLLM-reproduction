"""大模型 L（论文：One Fits All = GPT4TS，NeurIPS 2023，冻结自注意力与 FFN）。

架构（对齐论文 D 节 + 表 I + One Fits All 官方 GPT4TS 结构）:
  x ∈ R^{50×14} ──ReplicationPad1d 尾补 stride──▶ 13 个 patch（通道拼接，56 维）
    ──▶ TokenEmbedding（Conv1d 56→768, k=3, circular pad, kaiming）──▶ (B,13,768)
    ──▶ + 固定正弦位置嵌入 ──▶ GPT-2 N 层块（inputs_embeds 路径，内部加可训练 wpe）
    ──▶ φl(x) ∈ R^{13×768} ──▶ 预测头（末端 patch，MLP）──▶ yl ∈ R

阶段2：仅训练 patch 嵌入、位置嵌入(wpe)、LN 与预测头（冻结 attention+FFN，
OFA 官方：'ln' 与 'wpe' 可训练）。

关键修正（2026-08-29 复现突破）:
  - 输入编码由「Linear(56→768) + wpe 初始化可学习位置」改为 One Fits All 官方的
    「Conv1d TokenEmbedding + 固定正弦位置嵌入」——FD001 LM test 14.309 → 13.45
    （12 层，详见 experiments/ 与 docs/REVISIONS.md 新增条目）。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model

from ..config import LargeModelConfig


class TokenEmbedding(nn.Module):
    """OFA 官方 TokenEmbedding：Conv1d 值嵌入（kernel 3 + circular pad + kaiming）。"""

    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.token_conv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                    kernel_size=3, padding=1, padding_mode="circular",
                                    bias=False)
        nn.init.kaiming_normal_(self.token_conv.weight, mode="fan_in",
                                nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_patch, c_in) -> Conv1d over n_patch -> (B, d_model, n_patch)
        return self.token_conv(x.permute(0, 2, 1)).transpose(1, 2)


class PositionalEmbedding(nn.Module):
    """OFA 官方固定正弦位置嵌入（sin/cos，max_len=5000，不参与训练）。"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class LargeModel(nn.Module):
    def __init__(self, cfg: LargeModelConfig, n_sensors: int = 14, window: int = 50):
        super().__init__()
        self.cfg = cfg
        self.patch_size = cfg.patch_size
        self.stride = cfg.patch_stride
        # 13 个 patch：OFA 官方 patch_num = (seq-patch)//stride + 1，ReplicationPad 后再 +1
        self.n_patch = (window - cfg.patch_size) // cfg.patch_stride + 1 + 1

        # GPT-2 backbone（本地权重），随后冻结
        self.gpt2 = GPT2Model.from_pretrained(cfg.model_name)
        if cfg.n_blocks < len(self.gpt2.h):
            self.gpt2.h = self.gpt2.h[:cfg.n_blocks]
        # 冻结范围（OFA 官方）：仅 'ln' 与 'wpe' 可训练，attention+FFN 冻结
        for name, param in self.gpt2.named_parameters():
            if "ln" in name or "wpe" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # patch 嵌入：Conv1d TokenEmbedding（OFA 官方，非 Linear）
        in_dim = cfg.patch_size * n_sensors
        self.token_embed = TokenEmbedding(in_dim, cfg.d_embed)
        # 固定正弦位置嵌入（OFA 官方，非 wpe）
        self.pos_embed = PositionalEmbedding(cfg.d_embed)
        self.dropout = nn.Dropout(0.1)

        # 预测头：末端 patch 特征 → RUL（MLP，实测最优）
        if cfg.head_hidden:
            self.predictor = nn.Sequential(
                nn.Linear(cfg.d_embed, cfg.head_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(cfg.head_hidden, 1),
            )
        else:
            self.predictor = nn.Linear(cfg.d_embed, 1)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) → (B, n_patch, patch_size*C)。ReplicationPad1d 尾补（OFA 官方）。"""
        x = x.permute(0, 2, 1)                          # (B, C, T)
        x = F.pad(x, (0, self.stride), mode="replicate")  # (B, C, T+stride)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        # (B, C, n_patch, patch_size)
        B, C = x.shape[0], x.shape[1]
        x = x.reshape(B, C, self.n_patch, self.patch_size).permute(0, 2, 1, 3)
        return x.reshape(B, self.n_patch, self.patch_size * C)

    def forward(self, x: torch.Tensor, return_feat: bool = True):
        """x: (B, T, 14) → (yl: (B,), φl: (B, n_patch, 768))"""
        x = self._patchify(x)                          # (B, 13, 56)
        h = self.token_embed(x)                        # (B, 13, 768)
        h = h + self.pos_embed(h)                      # 固定正弦位置嵌入
        h = self.dropout(h)
        # inputs_embeds 路径（GPT2Model 内部自动加可训练 wpe，OFA 官方行为）
        h = self.gpt2(inputs_embeds=h).last_hidden_state  # (B, 13, 768)
        feat = h
        yl = self.predictor(h[:, -1])                  # 末端 patch（实测最优）
        if return_feat:
            return yl.squeeze(-1), feat
        return yl.squeeze(-1)


def trainable_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

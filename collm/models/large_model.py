"""大模型 L（论文：预训练 GPT-2 + patch embedding，冻结自注意力与 FFN）。

架构（对齐论文 D 节 + 表 I）:
  x ∈ R^{50×14} ──patch 化（patch=4, stride=4 → 12 个 patch）
    ──▶ Linear(4×14 → 768)  ──加可学习位置嵌入（wpe 前 12 位初始化）──▶
    ──▶ GPT-2 N 层块（冻结 attention+FFN；LN 微调）+ ln_f ──▶ φl(x) ∈ R^{12×768}
    ──▶ LM 预测头（末端 patch 特征，MLP）──▶ yl ∈ R

阶段2：仅训练 patch embedding、位置嵌入、LN 与预测头（论文：注意力与 FFN 全程冻结）。
层数：FD001=9、FD003=12（get_config 定稿，REVISIONS #27）。
"""
import torch
import torch.nn as nn
from transformers import GPT2Model

from ..config import LargeModelConfig


class LargeModel(nn.Module):
    def __init__(self, cfg: LargeModelConfig, n_sensors: int = 14):
        super().__init__()
        self.cfg = cfg
        # GPT-2 backbone（本地权重），随后冻结
        self.gpt2 = GPT2Model.from_pretrained(cfg.model_name)
        # 层数裁剪（FD001 前 9 层 / FD003 12 层）
        if cfg.n_blocks < len(self.gpt2.h):
            self.gpt2.h = self.gpt2.h[:cfg.n_blocks]
        if cfg.freeze_backbone:
            # 论文 D 节：保留"自注意力机制和前馈神经网络模块"不更新——冻结范围仅限
            # attention+FFN；逐层 LN（ln_1/ln_2）与 ln_f 可微调（论文字面对齐）
            for p in self.gpt2.parameters():
                p.requires_grad = False
            for block in self.gpt2.h:
                for p in block.ln_1.parameters():
                    p.requires_grad = cfg.learnable_ln_f
                for p in block.ln_2.parameters():
                    p.requires_grad = cfg.learnable_ln_f
        for p in self.gpt2.ln_f.parameters():
            p.requires_grad = cfg.learnable_ln_f

        # patch embedding：patch 化（无重叠，stride == patch_size）
        in_dim = cfg.patch_size * n_sensors
        self.patch_embed = nn.Linear(in_dim, cfg.d_embed)
        # 位置嵌入：可学习，从预训练 wpe 前 n_patch 位初始化（实测最优，REVISIONS #4/#17）
        self.pos_embed = nn.Parameter(
            self.gpt2.wpe.weight[:cfg.max_patches].unsqueeze(0).clone().detach())
        for p in self.gpt2.wpe.parameters():
            p.requires_grad = False
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
        """x: (B, T, C) → patches (B, n_patch, patch_size*C)。尾部不足直接截断。"""
        ps, st = self.cfg.patch_size, self.cfg.patch_stride
        pat = x.unfold(1, ps, st)                  # (B, n, C, ps)
        pat = pat.permute(0, 1, 3, 2).reshape(x.shape[0], -1, ps * x.shape[2])
        return pat

    def forward(self, x: torch.Tensor, return_feat: bool = True):
        """x: (B, T, 14) → (yl: (B,), φl: (B, n_patch, 768))"""
        x = self._patchify(x)                     # (B, n, 56)
        x = self.patch_embed(x)                   # (B, n, 768)
        x = x + self.pos_embed[:, : x.shape[1]]   # 可学习位置嵌入
        # 手动遍历 GPT-2 块（不使用 inputs_embeds 路径，避免其内部重复加 wpe）
        h = x
        for block in self.gpt2.h:
            h = block(h)
        out = self.gpt2.ln_f(h)                   # (B, n, 768)
        feat = out[:, -1]                         # 末端 patch（实测最优）
        yl = self.predictor(feat)                 # → (B, 1)
        if return_feat:
            return yl.squeeze(-1), out
        return yl.squeeze(-1)


def trainable_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

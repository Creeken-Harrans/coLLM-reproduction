"""大模型 L（论文：预训练 GPT-2 + patch embedding，冻结自注意力与 FFN）。

架构（对齐论文）:
  x ∈ R^{50×14} ──patch 化（patch=4, stride=4 → 12 个 patch）
    ──▶ Linear(4×14 → 768)  ──加位置嵌入（GPT-2 wpe，冻结）──▶
    ──▶ GPT-2 12 层块（冻结）+ ln_f ──▶ φl(x) ∈ R^{12×768}
    ──▶ LM Predictor（末端 patch 特征）──▶ yl ∈ R

阶段2：仅训练 patch embedding 与预测头（论文：注意力与 FFN 全程冻结）。
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
        # 层数裁剪：One Fits All 官方用前 6 层（"GPT-2 with 6 layers was found to be
        # a reasonable choice"）；论文表 II 的 One Fits All 基线按官方实现
        if cfg.n_blocks < len(self.gpt2.h):
            self.gpt2.h = self.gpt2.h[:cfg.n_blocks]
        if cfg.dropout != 0.1:
            # 覆盖 GPT-2 内部 dropout（实验开关）
            for m in self.gpt2.modules():
                if isinstance(m, nn.Dropout):
                    m.p = cfg.dropout
        if cfg.freeze_backbone:
            # 论文 D 节：保留"自注意力机制和前馈神经网络模块"不更新——冻结范围仅限
            # attention+FFN。learnable_ln_f=True 时各层 LayerNorm（ln_1/ln_2）与
            # ln_f 可微调（论文字面对齐）；False 时全部冻结（FD003 实测指标更好）。
            for p in self.gpt2.parameters():
                p.requires_grad = False
            for block in self.gpt2.h:
                for p in block.ln_1.parameters():
                    p.requires_grad = cfg.learnable_ln_f
                for p in block.ln_2.parameters():
                    p.requires_grad = cfg.learnable_ln_f
        gpt2_cfg = self.gpt2.config

        # patch embedding：patch 化（无重叠，stride == patch_size）
        in_dim = cfg.patch_size * n_sensors
        self.patch_embed = nn.Linear(in_dim, cfg.d_embed)
        if cfg.input_norm:
            # BANjian16 CoLLM 复现：proj → LayerNorm → GPT-2
            self.input_norm = nn.LayerNorm(cfg.d_embed)
        else:
            self.input_norm = nn.Identity()
        if cfg.channel_independent:
            # GPT4TS 官方：每通道独立 patch（4 步/通道），共享 GPT-2 权重
            self.patch_embed_ci = nn.Linear(cfg.patch_size, cfg.d_embed)
        # 位置嵌入：可学习。'wpe'=从预训练 wpe 前 n_patch 位初始化；
        # 'random'=std 0.02 随机（One Fits All 原版做法，REVISIONS #17）
        if cfg.pos_init == "wpe":
            # 与 random 分支同形状 (1, max_patches, d_embed)，forward 统一切片
            self.pos_embed = nn.Parameter(
                self.gpt2.wpe.weight[:cfg.max_patches].unsqueeze(0).clone().detach())
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, cfg.max_patches, cfg.d_embed))
            nn.init.normal_(self.pos_embed, std=0.02)
        if cfg.freeze_pos:
            self.pos_embed.requires_grad = False
        for p in self.gpt2.wpe.parameters():
            p.requires_grad = False
        # ln_f（final layernorm）不属于注意力/FFN，论文冻结范围未包含 → 可选微调
        for p in self.gpt2.ln_f.parameters():
            p.requires_grad = cfg.learnable_ln_f
        # 预测头：末端 patch 特征 → RUL（GPT4TS 官方为全 patch 拼接 → 单值）
        if cfg.head_concat:
            # lazy：实际 patch 数由窗口/填充决定（50→13），首轮 forward 构建
            self.predictor = None
        elif cfg.head_hidden:
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
        if self.cfg.channel_independent:
            return self._forward_ci(x, return_feat)
        if self.cfg.instance_norm:
            # One Fits All 核心组件：per-sample per-channel 归一化（μ/σ 取窗口内统计）
            mu = x.mean(dim=1, keepdim=True)
            sd = x.std(dim=1, keepdim=True).clamp(min=1e-5)
            x = (x - mu) / sd
        if self.cfg.pad_patches:
            # 窗口 50 / patch 4 = 12.5 → 补足 13 个完整 patch（GPT4TS 惯例）
            n_pad = self.cfg.patch_size - (x.shape[1] % self.cfg.patch_size)
            if n_pad < self.cfg.patch_size:
                if self.cfg.pad_mode == "zero":
                    x = torch.nn.functional.pad(x, (0, 0, 0, n_pad), mode="constant", value=0.0)
                else:
                    x = torch.cat([x, x[:, -1:].expand(-1, n_pad, -1)], dim=1)
        x = self._patchify(x)                     # (B, n, 56)
        x = self.input_norm(self.patch_embed(x))  # (B, n, 768)（可选 proj 后 LN）
        x = x + self.pos_embed[:, : x.shape[1]]   # 可学习位置嵌入
        # 手动遍历 GPT-2 块（不使用 inputs_embeds 路径，避免其内部重复加 wpe）
        h = x
        for block in self.gpt2.h:
            h = block(h)
        out = self.gpt2.ln_f(h)                   # (B, n, 768)
        if self.cfg.pool_mode == "mean":
            feat = out.mean(dim=1)                # 全部 patch 均值（实验变体）
        else:
            feat = out[:, -1]                     # 末端 patch（论文/One Fits All）
        if self.cfg.head_concat:
            if self.predictor is None:
                self.predictor = nn.Linear(out.shape[-1] * out.shape[-2], 1).to(out.device)
            yl = self.predictor(out.reshape(x.shape[0], -1))
        else:
            yl = self.predictor(feat)
        if return_feat:
            return yl.squeeze(-1), out
        return yl.squeeze(-1)

    def _forward_ci(self, x: torch.Tensor, return_feat: bool = True):
        """GPT4TS 官方风格：通道独立。x: (B, T, C) → 每通道独立 patch+GPT-2。

        ReplicationPad 尾补 stride 步 → (B*C, n_patch, patch_size) → 共享
        GPT-2 → 每通道预测 RUL → 14 通道平均。φl 返回 14 通道特征均值。
        """
        B, T, C = x.shape
        if self.cfg.instance_norm:
            mu = x.mean(dim=1, keepdim=True)
            sd = x.std(dim=1, keepdim=True).clamp(min=1e-5)
            x = (x - mu) / sd
        x = x.permute(0, 2, 1)                    # (B, C, T)
        # ReplicationPad1d 尾补 stride 步（GPT4TS 官方），50+4=54 → 13 patch
        x = torch.nn.functional.pad(x, (0, self.cfg.patch_stride), mode="replicate")
        pat = x.unfold(2, self.cfg.patch_size, self.cfg.patch_stride)   # (B, C, n, ps)
        pat = pat.reshape(B * C, -1, self.cfg.patch_size)
        h = self.input_norm(self.patch_embed_ci(pat))  # (B*C, n, 768)
        h = h + self.pos_embed[:, : h.shape[1]]
        for block in self.gpt2.h:
            h = block(h)
        out = self.gpt2.ln_f(h)                   # (B*C, n, 768)
        if self.cfg.pool_mode == "mean":
            feat = out.mean(dim=1)
        else:
            feat = out[:, -1]
        if self.cfg.head_concat:
            if self.predictor is None:
                self.predictor = nn.Linear(out.shape[-1] * out.shape[-2], 1).to(out.device)
            yl = self.predictor(out.reshape(B * C, -1))   # (B*C, 1)
        else:
            yl = self.predictor(feat)             # (B*C, 1)
        yl = yl.reshape(B, C).mean(dim=1)         # 14 通道平均 → (B,)
        if return_feat:
            # φl：14 通道特征均值（B, n, 768），供反思网络（temporal flatten）
            feat_mean = out.reshape(B, C, out.shape[1], -1).mean(dim=1)
            return yl, feat_mean
        return yl


def trainable_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

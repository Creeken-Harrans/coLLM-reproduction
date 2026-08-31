"""忠实复刻 GPT4TS（One Fits All, NeurIPS 2023）用于 CMAPSS RUL 标量回归。

背景：论文 CoLLM 表 II 的 LM = "One Fits All Fine-tuning"（FD001 12.34 / FD003 11.18），
且正文 B 节明确"使用 One Fits All 作为大模型实现 CoLLM"。One Fits All 官方代码
(DAMO-DI-ML/NeurIPS2023-One-Fits-All) 的 GPT4TS 结构（Classification/src/models/gpt4ts.py，
与标量输出任务同构）关键点：

  - 输入 patch 化：ReplicationPad1d 尾补 stride → (B, M, L+stride) → unfold →
    (B, n_patch, patch_size*M)（通道拼接），窗口 50 / patch 4 / stride 4 → 13 patch × 56
  - TokenEmbedding：Conv1d(56→768, kernel=3, padding=1, padding_mode='circular', bias=False)
    kaiming 初始化（非 Linear）
  - PositionalEmbedding：固定正弦（DataEmbedding 内），可学习/禁用为可选
  - GPT-2：冻结 self-attention + FFN，仅 'ln' 与 'wpe' 可训练；用 inputs_embeds 路径
    （GPT2Model 内部会自动加 wpe）
  - 输出头：GELU → 展平全部 patch → LayerNorm(768*n_patch) → Linear(768*n_patch → 1)

与复现定稿 LargeModel 的关键差异（本脚本逐项对照）：
  1) patch 嵌入：Conv1d TokenEmbedding vs Linear(56→768)
  2) 位置嵌入：正弦/可学习 vs 从 GPT-2 wpe 初始化
  3) 输出头：全 patch GELU+LN+Linear vs 末端 patch MLP
  4) padding：13 patch 复制尾补 vs 12 patch 截断
  5) 层数：可扫（OFA 默认 6；论文 FLOPs 2.21G 反推 12）

防泄漏纪律：数据划分 seed 固定 42；μ/σ 仅 train；val 按 unit 20%；选择只用 val，
test 仅对照报告；不 test 挑选。

用法:
  python experiments/gpt4ts_faithful.py --subset FD001 \
      --layers 6 --instancenorm 0 --pos sinusoidal --lr 5e-4 --batch 256
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2Model

from collm.config import get_config
from collm.data import prepare_cmapss
from collm.training import set_seed, rmse_mae, evaluate


# ---------------------------------------------------------------------------
# 官方 embed.py 逐字复刻（PositionalEmbedding / TokenEmbedding）
# ---------------------------------------------------------------------------
class PositionalEmbedding(nn.Module):
    """固定正弦位置嵌入（OFA 官方 embed.py 逐字）。"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(np.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    """OFA 官方 TokenEmbedding：Conv1d 值嵌入，kaiming 初始化。"""

    def __init__(self, c_in, d_model):
        super().__init__()
        padding = 1
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding,
                                   padding_mode="circular", bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in",
                                        nonlinearity="leaky_relu")

    def forward(self, x):
        # x: (B, n_patch, c_in) -> Conv1d over n_patch
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


# ---------------------------------------------------------------------------
# 忠实 GPT4TS（分类式，标量回归输出）
# ---------------------------------------------------------------------------
class GPT4TS_RUL(nn.Module):
    def __init__(self, cfg, n_sensors=14, layers=6, instancenorm=False,
                 pos_mode="sinusoidal", head="concat", embed="conv"):
        super().__init__()
        self.patch_size = cfg.large.patch_size        # 4
        self.stride = cfg.large.patch_stride          # 4
        self.d_model = cfg.large.d_embed              # 768
        self.n_sensors = n_sensors                    # 14
        self.instancenorm = instancenorm
        self.pos_mode = pos_mode
        self.head = head
        self.embed = embed

        seq_len = cfg.data.window                       # 50
        self.patch_num = (seq_len - self.patch_size) // self.stride + 1  # 12
        self.patch_num += 1                             # 13（ReplicationPad 后）

        # GPT-2 主干（先加载，wpe 初始化位置嵌入需要）
        self.gpt2 = GPT2Model.from_pretrained(cfg.large.model_name)
        if layers < len(self.gpt2.h):
            self.gpt2.h = self.gpt2.h[:layers]
        # 冻结：仅 'ln' 与 'wpe' 可训练（OFA 官方）
        for name, param in self.gpt2.named_parameters():
            if "ln" in name or "wpe" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # patch 嵌入：Conv1d TokenEmbedding（OFA 官方）或 Linear（复现定稿）
        if embed == "conv":
            self.token_embed = TokenEmbedding(self.patch_size * n_sensors,
                                              self.d_model)
        else:
            self.token_embed = nn.Linear(self.patch_size * n_sensors, self.d_model)
        # 位置嵌入：正弦（固定）/ 可学习 / wpe（复现定稿）/ 禁用
        if pos_mode == "sinusoidal":
            self.pos_embed = PositionalEmbedding(self.d_model)
        elif pos_mode == "learnable":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_num,
                                                      self.d_model))
            nn.init.normal_(self.pos_embed, std=0.02)
        elif pos_mode == "wpe":
            self.pos_embed = nn.Parameter(
                self.gpt2.wpe.weight[:self.patch_num].unsqueeze(0).clone().detach())
        else:  # none
            self.pos_embed = None
        self.dropout = nn.Dropout(0.1)

        # 输出头：concat（OFA 官方）或 last-patch MLP（复现定稿）
        if head == "concat":
            self.ln_proj = nn.LayerNorm(self.d_model * self.patch_num)
            self.out_layer = nn.Linear(self.d_model * self.patch_num, 1)
        else:  # last-patch MLP（与复现 LargeModel 同构）
            self.predictor = nn.Sequential(
                nn.Linear(self.d_model, 128), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(128, 1))

    def forward(self, x, return_feat=False):
        B, L, M = x.shape
        if self.instancenorm:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True,
                                         unbiased=False) + 1e-5).detach()
            x = x / stdev

        # patch 化（OFA 官方：ReplicationPad1d 尾补 + unfold + 通道拼接）
        x = x.permute(0, 2, 1)                                   # (B, M, L)
        x = F.pad(x, (0, self.stride), mode="replicate")         # (B, M, L+stride)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        # (B, M, n_patch, patch_size)
        x = x.reshape(B, M, -1, self.patch_size).permute(0, 2, 1, 3)
        x = x.reshape(B, self.patch_num, self.patch_size * M)    # (B, 13, 56)

        # TokenEmbedding + 位置嵌入 + dropout
        h = self.token_embed(x)                                  # (B, 13, 768)
        if self.pos_mode in ("learnable", "wpe"):
            h = h + self.pos_embed[:, : self.patch_num]
        elif self.pos_embed is not None:
            h = h + self.pos_embed(h)
        h = self.dropout(h)

        # GPT-2（inputs_embeds 路径，内部自动加 wpe）
        h = self.gpt2(inputs_embeds=h).last_hidden_state        # (B, 13, 768)

        feat = h
        # 输出头：concat（GELU→展平→LN→Linear）或 last-patch MLP
        if self.head == "concat":
            out = F.gelu(h).reshape(B, -1)
            out = self.ln_proj(out)
            y = self.out_layer(out).squeeze(-1)
        else:
            y = self.predictor(h[:, -1]).squeeze(-1)
        if return_feat:
            return y, feat
        return y


def trainable_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001", choices=["FD001", "FD003"])
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--instancenorm", type=int, default=0, choices=[0, 1])
    ap.add_argument("--pos", default="sinusoidal",
                    choices=["sinusoidal", "learnable", "wpe", "none"])
    ap.add_argument("--head", default="concat", choices=["concat", "last"])
    ap.add_argument("--embed", default="conv", choices=["conv", "linear"])
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--sched", default="none", choices=["none", "cosine"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = get_config(args.subset)
    set_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    tag = (f"{args.subset}_L{args.layers}_IN{args.instancenorm}_"
           f"pos{args.pos}_{args.head}_{args.embed}_lr{args.lr}_b{args.batch}")

    tr_ds, val_ds, te_ds, stats = prepare_cmapss(cfg.data, args.subset, cfg.seed)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=2,
                       pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False)
    te_ld = DataLoader(te_ds, batch_size=args.batch, shuffle=False)

    model = GPT4TS_RUL(cfg, n_sensors=cfg.data.n_sensors, layers=args.layers,
                       instancenorm=bool(args.instancenorm),
                       pos_mode=args.pos, head=args.head,
                       embed=args.embed).to(device)
    n_tr = trainable_params(model)
    n_tot = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] 可训练 {n_tr/1e3:.1f}K / 总 {n_tot/1e6:.1f}M | "
          f"train {stats['train_windows']} val {stats['val_windows']} "
          f"test {stats['test_windows']}", flush=True)

    optimizer = torch.optim.Adam([p for p in model.parameters()
                                  if p.requires_grad], lr=args.lr)
    scheduler = None
    if args.sched == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    criterion = nn.MSELoss()
    best_val, best_ep = float("inf"), -1
    best_state = None
    hist = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in tr_ld:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        val_rmse, val_mae = evaluate(model, val_ld, device)
        hist.append({"epoch": epoch, "val_rmse": val_rmse})
        improved = val_rmse < best_val
        if improved:
            best_val, best_ep = val_rmse, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or improved:
            print(f"  epoch {epoch:3d} val RMSE {val_rmse:.4f}"
                  + ("  *" if improved else ""), flush=True)
        if epoch - best_ep >= args.patience:
            print(f"  早停 @epoch {epoch}（最优 {best_ep}）", flush=True)
            break

    model.load_state_dict(best_state)
    te_rmse, te_mae = evaluate(model, te_ld, device)
    print(f"[{tag}] BEST val RMSE {best_val:.3f} @ep{best_ep} | "
          f"TEST RMSE {te_rmse:.3f} MAE {te_mae:.3f}", flush=True)

    # 记录（写入 worktree 内 experiments/results/，不入库主仓）
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = {"tag": tag, "subset": args.subset, "layers": args.layers,
           "instancenorm": args.instancenorm, "pos": args.pos, "lr": args.lr,
           "batch": args.batch, "epochs": args.epochs, "seed": args.seed,
           "best_val_rmse": best_val, "best_epoch": best_ep,
           "test_rmse": te_rmse, "test_mae": te_mae, "history": hist}
    (out_dir / f"gpt4ts_{tag}.json").write_text(
        json.dumps(rec, indent=1, ensure_ascii=False))
    print(f"  记录: {out_dir / f'gpt4ts_{tag}.json'}", flush=True)


if __name__ == "__main__":
    main()

"""FLOPs 估计（论文：FLOPs 衡量单次推理浮点运算数，与运行环境无关）。

口径：MAC×2（每次乘-加计 2 FLOPs），覆盖:
  - nn.Linear / transformers GPT2Conv1D  → 2·T·in·out
  - nn.LayerNorm / GPT2LayerNorm         → ~3·numel
  - nn.MultiheadAttention（SM 的 Encoder）→ 4·T²·d（QK^T 与 @V）
  - GPT2Attention                        → 4·T²·n_embd（同理由）
  - GELU                                 → ~10·numel（近似）
sigmoid、tanh、Dropout、unfold 忽略（占比极小）。

加速比（论文口径）:
  speedup = FLOPs_LM / (FLOPs_SM + FLOPs_FNN + p×(FLOPs_LM + FLOPs_R))
其中 p = 触发大模型的样本比例（来自实际推理路由统计）。
"""
import torch
import torch.nn as nn


def module_flops(model: nn.Module, x: torch.Tensor, extra=None) -> float:
    """对给定输入计算模型总 FLOPs（x 含 batch 维，按 batch=1 折算）。"""
    total = 0.0
    hooks = []

    def hook_fn(m):
        def fn(_m, inp, _out):
            nonlocal total
            if not inp:
                return
            i = inp[0] if isinstance(inp, (tuple, list)) else inp
            if i is None or not hasattr(i, "ndim") or i.ndim < 2:
                return
            B, T, C = i.shape[0], i.shape[-2] if i.ndim >= 3 else 1, i.shape[-1]
            if isinstance(_m, nn.Linear):
                total += 2 * T * _m.in_features * _m.out_features
            elif _m.__class__.__name__ in ("GPT2Conv1D", "Conv1D"):
                # transformers GPT-2 的 1×1 Conv1D：weight (in, out)
                w = _m.weight
                total += 2 * T * w.shape[0] * w.shape[1]
            elif isinstance(_m, nn.LayerNorm) or _m.__class__.__name__ == "GPT2LayerNorm":
                total += 3 * torch.tensor(_m.normalized_shape).prod().item()
            elif isinstance(_m, nn.MultiheadAttention):
                total += 4 * T * T * _m.embed_dim
            elif _m.__class__.__name__ == "GPT2Attention":
                embed = getattr(_m, "embed_dim", getattr(_m, "nx", 768))
                total += 4 * T * T * embed
            elif isinstance(_m, nn.GELU):
                total += 10 * torch.tensor(i.shape[1:]).prod().item()
        return fn

    for m in model.modules():
        h = m.register_forward_hook(hook_fn(m))
        hooks.append(h)

    with torch.no_grad():
        model(x, extra) if extra is not None else model(x)

    for h in hooks:
        h.remove()
    return total


def collm_flops(module, x: torch.Tensor) -> dict:
    """分别统计 SM、FNN、LM、R 的 FLOPs。x: (B, T, C)。"""
    from .models.collm import CoLLM
    assert isinstance(module, CoLLM)
    out = {}
    with torch.no_grad():
        out["small"] = module_flops(module.small, x)
        ys, feat_s = module.small(x)
        out["fuzzy"] = module_flops(module.fuzzy, feat_s, ys if module.cfg.fuzzy.cat_pred else None)
        out["large"] = module_flops(module.large, x)
        yl, feat_l = module.large(x)
        out["reflection"] = module_flops(module.reflection, feat_l, yl if module.cfg.reflection.cat_pred else None)
    return out


def collm_speedup(flops: dict, lm_ratio: float) -> float:
    """论文口径的 FLOPs 加速比。lm_ratio = 触发大模型的样本比例 p。"""
    lm = flops["large"]
    collm = flops["small"] + flops["fuzzy"] + lm_ratio * (lm + flops["reflection"])
    return lm / collm

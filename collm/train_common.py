"""训练通用工具：种子、评估、早停训练循环、checkpoint。"""
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rmse_mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> tuple[float, float]:
    err = y_pred - y_true
    rmse = float(torch.sqrt((err ** 2).mean()).item())
    mae = float(err.abs().mean().item())
    return rmse, mae


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str,
             featurize=None) -> tuple[float, float]:
    """返回 (RMSE, MAE)。featurize: 可选，将 loader 输出转为 (x, y)。"""
    model.eval()
    preds, trues = [], []
    for batch in loader:
        x, y = batch if featurize is None else featurize(batch)
        x, y = x.to(device), y.to(device)
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        preds.append(out.detach().cpu())
        trues.append(y.cpu())
    preds = torch.cat(preds)
    trues = torch.cat(trues)
    return rmse_mae(preds, trues)


def train_loop(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
               optimizer, criterion, device: str, epochs: int, patience: int,
               tag: str, ckpt_path: Path, metrics: dict | None = None,
               max_batches: int | None = None, scheduler=None,
               logger: logging.Logger | None = None,
               save_history: bool = True) -> float:
    """早停训练循环，保存最优 val RMSE 权重。返回最优验证 RMSE。

    - 训练历史（每 epoch train_loss/val_rmse/val_mae）保存为 ckpt 同目录 history JSON；
    - 早停同时记录 val 最优与训练结束两个 checkpoint。
    """
    ckpt_path = Path(ckpt_path)
    best_val = float("inf")
    best_epoch = -1
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    hist = []
    log = logger if logger is not None else logging.getLogger(tag)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for i, batch in enumerate(train_loader):
            x, y = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * len(y)
            n += len(y)
            if max_batches and i + 1 >= max_batches:
                break
        if scheduler is not None:
            scheduler.step()

        val_rmse, val_mae = evaluate(model, val_loader, device)
        hist.append({"epoch": epoch, "train_loss": total_loss / max(1, n),
                     "val_rmse": val_rmse, "val_mae": val_mae})
        improved = val_rmse < best_val
        if improved:
            best_val = val_rmse
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)
        if epoch % 5 == 0 or improved or epoch == epochs:
            log.info(f"epoch {epoch:3d} | loss {hist[-1]['train_loss']:.4f} "
                     f"| val RMSE {val_rmse:.4f} MAE {val_mae:.4f}"
                     + ("  *" if improved else ""))
        if epoch - best_epoch >= patience:
            log.info(f"早停于 epoch {epoch}（最优 epoch {best_epoch}，val RMSE {best_val:.4f}）")
            break

    # 训练历史持久化（防丢失 + 曲线绘制）
    if save_history:
        hist_path = ckpt_path.with_name(ckpt_path.stem + "_history.json")
        hist_path.write_text(json.dumps(hist, ensure_ascii=False, indent=1))
    if metrics is not None:
        metrics[f"{tag}_history"] = hist
    return best_val

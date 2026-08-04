"""阶段3：训练置信度模块（FNN 决策智能体 + 自反思 FCN）。大小模型全部冻结。

论文: Q*_s = 1 - tanh(|y_s - y*|/α)，Q*_l = 1 - tanh(|y_l - y*|/α)；
      FNN: φs(x) → Q_s；FCN: φl(x) 展平 + 单层全连接 → Q_l；均 MSE 训练。
先冻结模型预提取特征，再快速训练两个置信度网络（联合优化）。

用法: python scripts/train_conf.py --subset FD001 [--lr 2e-3 --epochs 100]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from collm.config import Config
from collm.data import prepare_cmapss
from collm.logging_utils import setup_logger, log_metrics
from collm.models.collm import CoLLM
from collm.models.fuzzy import FuzzyAgent, confidence_label
from collm.models.reflection import ReflectionModel
from collm.train_common import set_seed


@torch.no_grad()
def extract_features(model: CoLLM, loader: DataLoader, device: str, alpha: float, alpha_l: float | None = None):
    """预提取 (φs, φl, ys, yl, y*)，并构造两个置信度标签。"""
    fs, fl, qs_t, ql_t, ys, yl, yt = [], [], [], [], [], [], []
    model.eval()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        ys_b, feat_s = model.small(x)
        yl_b, feat_l = model.large(x)
        qs_b = confidence_label(ys_b, y, alpha)
        ql_b = confidence_label(yl_b, y, alpha_l or alpha)
        fs.append(feat_s.cpu()); fl.append(feat_l.cpu())
        qs_t.append(qs_b.cpu()); ql_t.append(ql_b.cpu())
        ys.append(ys_b.cpu()); yl.append(yl_b.cpu()); yt.append(y.cpu())
    return (torch.cat(fs), torch.cat(fl), torch.cat(qs_t), torch.cat(ql_t),
            torch.cat(ys), torch.cat(yl), torch.cat(yt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001", choices=["FD001", "FD003"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3, help="阶段3 lr（2e-3 偏大，FNN 早停于 epoch 1）")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alpha", type=float, default=None, help="残差缩放系数 α（FNN 置信度标签 Q*s）")
    ap.add_argument("--alpha-l", type=float, default=None, help="反思网络标签 Q*l 的 α（默认=α；α5 使 Ql 塌缩低端，图5 LM 分箱空）")
    ap.add_argument("--feat-mode", default="fuzzy", choices=["fuzzy", "raw"],
                    help="FNN 输入：模糊特征（论文公式9/设计决策#17）或原始 φs 池化（公式11 字面，消融）")
    ap.add_argument("--hidden", type=int, default=None, help="FNN hidden 宽度（默认 cfg）")
    ap.add_argument("--pool-mode", default=None, choices=["mean", "stats", "flatten"],
                    help="时间聚合：mean / stats（mean+max+std+last）/ flatten（全步展平）")
    ap.add_argument("--blocks", type=int, default=None, help="LM 层数（FD001=9、FD003=12）")
    ap.add_argument("--ref-norm", action="store_true", help="反思网络输入 LayerNorm（默认关——论文字面单层全连接）")
    ap.add_argument("--cat-pred", action="store_true", help="FNN/反思输入拼接预测值 ys/yl（联合分布实验）")
    ap.add_argument("--fixed", action="store_true", help="固定 epochs 训练（不早停——网格实测更优）")
    ap.add_argument("--use-val-train", action="store_true", help="FNN/反思训练数据 = train+val 全部（网格实测更优）")
    args = ap.parse_args()

    cfg = Config()
    set_seed(cfg.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    alpha = args.alpha or cfg.fuzzy.alpha
    alpha_l = args.alpha_l or alpha
    if args.hidden is not None:
        cfg.fuzzy.hidden = args.hidden
    cfg.fuzzy.feat_mode = args.feat_mode
    if args.pool_mode is not None:
        cfg.fuzzy.pool_mode = args.pool_mode
    if args.blocks is not None:
        cfg.large.n_blocks = args.blocks
    cfg.reflection.use_norm = args.ref_norm
    cfg.fuzzy.cat_pred = args.cat_pred
    cfg.reflection.cat_pred = args.cat_pred
    # 反思输入维度 = LM 实际 patch 数 × 768（LM 结构决定，勿硬编码）
    n_patch = (cfg.data.window + cfg.large.patch_stride) // cfg.large.patch_size  # 50+4=54→13（GPT4TS pad）
    if not cfg.large.pad_patches:
        n_patch = cfg.data.window // cfg.large.patch_size
    cfg.reflection.d_input = n_patch * cfg.large.d_embed
    ckpt_dir = Path(cfg.out_dir) / "checkpoints" / args.subset
    # 从权重自动推断 SM/LM 层数（防配置与权重不一致）
    _sd_sm = torch.load(ckpt_dir / "small.pt", map_location="cpu")
    cfg.small.n_layers = max(int(k.split(".")[2]) for k in _sd_sm if "encoder.layers." in k) + 1
    _sd_lg = torch.load(ckpt_dir / "large.pt", map_location="cpu")
    cfg.large.n_blocks = max(int(k.split(".")[2]) for k in _sd_lg if "gpt2.h." in k) + 1

    # ---- 组装完整 CoLLM 并加载前两阶段权重 ----
    model = CoLLM(cfg).to(device)
    model.small.load_state_dict(torch.load(ckpt_dir / "small.pt", map_location=device))
    model.large.load_state_dict(torch.load(ckpt_dir / "large.pt", map_location=device))
    for m in (model.small, model.large):
        for p in m.parameters():
            p.requires_grad = False
    model.eval()

    tr_ds, val_ds, te_ds, stats = prepare_cmapss(cfg.data, args.subset, cfg.seed)
    log = setup_logger("stage3", args.subset)
    log.info("预提取特征（大小模型冻结）...")
    feats = {}
    for name, ds in [("train", tr_ds), ("val", val_ds), ("test", te_ds)]:
        ld = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=cfg.train.num_workers, pin_memory=True)
        feats[name] = extract_features(model, ld, device, alpha, alpha_l)
        fs, fl, _, _, _, _, _ = feats[name]
        log.info(f"  {name}: φs{tuple(fs.shape)} φl{tuple(fl.shape)}")

    # α 校准检查：Q* 标签分布（应覆盖 [0,1] 且不过度饱和）
    _, _, qs_t, ql_t, _, _, _ = feats["train"]
    log.info(f"α={alpha}: Q*_s 均值 {qs_t.mean():.3f} 分位 "
             f"[{qs_t.quantile(0.1):.2f}, {qs_t.median():.2f}, {qs_t.quantile(0.9):.2f}] | "
             f"Q*_l 均值 {ql_t.mean():.3f} 分位 "
             f"[{ql_t.quantile(0.1):.2f}, {ql_t.median():.2f}, {ql_t.quantile(0.9):.2f}]")

    # ---- FNN 与 R 联合训练（阶段3，其余冻结）----
    # FNN 分布感知初始化：按 SM 特征实际分布设置 μ/σ（REVISIONS #13）
    init_mu, init_sigma = FuzzyAgent.init_from_features(feats["train"][0], cfg.fuzzy.n_membership // cfg.fuzzy.d_input)
    fuzzy = FuzzyAgent(cfg.fuzzy, init_mu=init_mu, init_sigma=init_sigma).to(device)
    log.info(f"FNN 初始化: σ={init_sigma:.2f}, μ 范围 [{init_mu.min():.2f}, {init_mu.max():.2f}]")
    reflection = ReflectionModel(cfg.reflection).to(device)
    params = list(fuzzy.parameters()) + list(reflection.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    criterion = nn.MSELoss()

    def make_loader(name, batch=1024, shuffle=True):
        fs, fl, qs_t, ql_t, ys, yl, _ = feats[name]
        ds = TensorDataset(fs, fl, qs_t, ql_t, ys, yl)
        # num_workers=0：多 worker 的 shuffle 不受 set_seed 控制（阶段3 曾出现
        # 同配置两次结果不同——worker RNG 独立），确定性优先
        return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0)

    if args.use_val_train:
        fs, fl, qs_t, ql_t, ys, yl, yt = feats["train"]
        fsv, flv, qsv, qlv, ysv, ylv, ytv = feats["val"]
        feats["trainval"] = (torch.cat([fs, fsv]), torch.cat([fl, flv]),
                             torch.cat([qs_t, qsv]), torch.cat([ql_t, qlv]),
                             torch.cat([ys, ysv]), torch.cat([yl, ylv]), torch.cat([yt, ytv]))
        log.info("FNN/反思训练数据 = train+val 全部")
    best_val = float("inf")
    best_ep = -1
    ckpt_f = ckpt_dir / "fuzzy.pt"
    ckpt_r = ckpt_dir / "reflection.pt"
    val_ld = make_loader("val", shuffle=False)

    for epoch in range(1, args.epochs + 1):
        fuzzy.train(); reflection.train()
        total, n = 0.0, 0
        for fs, fl, qs_t, ql_t, ys_b, yl_b in make_loader("trainval" if args.use_val_train else "train"):
            fs, fl = fs.to(device), fl.to(device)
            qs_t, ql_t = qs_t.to(device), ql_t.to(device)
            ys_b, yl_b = ys_b.to(device), yl_b.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = (criterion(fuzzy(fs, ys_b if args.cat_pred else None), qs_t)
                    + criterion(reflection(fl, yl_b if args.cat_pred else None), ql_t))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(fs); n += len(fs)

        # 验证：置信度 MSE
        fuzzy.eval(); reflection.eval()
        vloss = 0.0; vn = 0
        with torch.no_grad():
            for fs, fl, qs_t, ql_t, ys_b, yl_b in val_ld:
                fs, fl = fs.to(device), fl.to(device)
                qs_t, ql_t = qs_t.to(device), ql_t.to(device)
                ys_b, yl_b = ys_b.to(device), yl_b.to(device)
                l = (criterion(fuzzy(fs, ys_b if args.cat_pred else None), qs_t)
                     + criterion(reflection(fl, yl_b if args.cat_pred else None), ql_t))
                vloss += l.item() * len(fs); vn += len(fs)
        vloss /= vn
        if vloss < best_val:
            best_val = vloss
            best_ep = epoch
            torch.save(fuzzy.state_dict(), ckpt_f)
            torch.save(reflection.state_dict(), ckpt_r)
        if epoch % 10 == 0 or epoch == args.epochs:
            log.info(f"epoch {epoch:3d} | loss {total/max(1,n):.5f} | val {vloss:.5f}")
        if (not args.fixed) and epoch - best_ep >= 15:
            log.info(f"早停于 epoch {epoch}（最优 {best_ep}）")
            break

    if args.fixed:
        torch.save(fuzzy.state_dict(), ckpt_f)
        torch.save(reflection.state_dict(), ckpt_r)
        log.info("固定训练完成，保存最终权重")
    log_metrics(log, stage="stage3", alpha=alpha, best_val_mse=best_val, best_epoch=best_ep)
    log.info(f"阶段3完成，最优 val MSE {best_val:.5f}")


if __name__ == "__main__":
    main()

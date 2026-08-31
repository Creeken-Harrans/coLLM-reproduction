"""阶段3：训练置信度模块（FNN 决策智能体 + 自反思 FCN）。大小模型全部冻结。

论文: Q*_s = 1 - tanh(|y_s - y*|/α)，Q*_l = 1 - tanh(|y_l - y*|/α)（公式 10/13，同一 α）；
      FNN: φs(x) → Q_s（公式 11）；FCN: φl(x) 展平 + 单层全连接 → Q_l（公式 12）；
      联合 MSE 训练（公式 14），R 与 F 学习输入特征与预测结果的联合分布。

配置按子集自动加载（collm.config.get_config）：
  FD001: stats 聚合、α=6 | FD003: mean 聚合、α=6
  训练数据 = train+val 全部窗口（浅层模块，无早停泄漏），固定 epochs（实测最优）

用法: python scripts/train/train_conf.py --subset FD001
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from collm.config import get_config
from collm.data import prepare_cmapss
from collm.logging import setup_logger, log_metrics
from collm.models.collm import CoLLM
from collm.models.fuzzy import FuzzyAgent, confidence_label
from collm.models.reflection import ReflectionModel
from collm.training import set_seed


@torch.no_grad()
def extract_features(model: CoLLM, loader: DataLoader, device: str,
                     alpha_s: float, alpha_l: float):
    """预提取 (φs, φl, ys, yl, y*)，并构造两个置信度标签（大小模型冻结）。

    alpha_s/alpha_l 分别来自 cfg.fuzzy.alpha 与 cfg.reflection.alpha
    （论文公式 10/13 同一符号 α，本实现两处显式取值，定稿 FD001=6 / FD003=6）。"""
    fs, fl, qs_t, ql_t, ys, yl, yt = [], [], [], [], [], [], []
    model.eval()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        ys_b, feat_s = model.small(x)
        yl_b, feat_l = model.large(x)
        qs_b = confidence_label(ys_b, y, alpha_s)
        ql_b = confidence_label(yl_b, y, alpha_l)
        fs.append(feat_s.cpu()); fl.append(feat_l.cpu())
        qs_t.append(qs_b.cpu()); ql_t.append(ql_b.cpu())
        ys.append(ys_b.cpu()); yl.append(yl_b.cpu()); yt.append(y.cpu())
    return (torch.cat(fs), torch.cat(fl), torch.cat(qs_t), torch.cat(ql_t),
            torch.cat(ys), torch.cat(yl), torch.cat(yt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001", choices=["FD001", "FD003"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = get_config(args.subset)
    set_seed(cfg.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    alpha_s = cfg.fuzzy.alpha
    alpha_l = cfg.reflection.alpha
    ckpt_dir = Path(cfg.out_dir) / "checkpoints" / args.subset

    # ---- 组装完整 CoLLM 并加载前两阶段权重（全冻结）----
    model = CoLLM(cfg).to(device)
    model.small.load_state_dict(torch.load(ckpt_dir / "small.pt", map_location=device))
    model.large.load_state_dict(torch.load(ckpt_dir / "large.pt", map_location=device))
    for m in (model.small, model.large):
        for p in m.parameters():
            p.requires_grad = False
    model.eval()

    tr_ds, val_ds, te_ds, stats = prepare_cmapss(cfg.data, args.subset, cfg.seed)
    log = setup_logger("stage3", args.subset)
    log.info(f"配置: {args.subset} | α_s={alpha_s} α_l={alpha_l} | FNN {cfg.fuzzy.pool_mode} 聚合单层头 "
             f"| 反思单层无 LN | 训练数据 = train+val | 固定 {cfg.train.conf_epochs} epochs")
    log.info("预提取特征（大小模型冻结）...")
    feats = {}
    for name, ds in [("train", tr_ds), ("val", val_ds), ("test", te_ds)]:
        ld = DataLoader(ds, batch_size=cfg.train.conf_batch, shuffle=False,
                        num_workers=0, pin_memory=True)   # num_workers=0：确定性
        feats[name] = extract_features(model, ld, device, alpha_s, alpha_l)
        fs, fl, _, _, _, _, _ = feats[name]
        log.info(f"  {name}: φs{tuple(fs.shape)} φl{tuple(fl.shape)}")

    # α 校准检查：Q* 标签分布（应覆盖 [0,1] 且不过度饱和）
    _, _, qs_t, ql_t, _, _, _ = feats["train"]
    log.info(f"Q*_s 均值 {qs_t.mean():.3f} 分位 "
             f"[{qs_t.quantile(0.1):.2f}, {qs_t.median():.2f}, {qs_t.quantile(0.9):.2f}] | "
             f"Q*_l 均值 {ql_t.mean():.3f} 分位 "
             f"[{ql_t.quantile(0.1):.2f}, {ql_t.median():.2f}, {ql_t.quantile(0.9):.2f}]")

    # ---- FNN 与 R 联合训练（阶段3，其余冻结）----
    # FNN 分布感知初始化：按 SM 特征实际分布设置 μ/σ（REVISIONS #13）
    init_mu, init_sigma = FuzzyAgent.init_from_features(
        feats["train"][0], cfg.fuzzy.n_membership // cfg.fuzzy.d_input)
    fuzzy = FuzzyAgent(cfg.fuzzy, init_mu=init_mu, init_sigma=init_sigma).to(device)
    reflection = ReflectionModel(cfg.reflection).to(device)
    params = list(fuzzy.parameters()) + list(reflection.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.train.conf_lr)
    criterion = nn.MSELoss()

    # 训练数据 = train+val 全部（阶段3 是浅层模块；val 不参与早停选择——固定 epochs）
    fs, fl, qs_t, ql_t, _, _, _ = feats["train"]
    fsv, flv, qsv, qlv, _, _, _ = feats["val"]
    ds = TensorDataset(torch.cat([fs, fsv]), torch.cat([fl, flv]),
                       torch.cat([qs_t, qsv]), torch.cat([ql_t, qlv]))
    loader = DataLoader(ds, batch_size=cfg.train.conf_batch, shuffle=True, num_workers=0)

    ckpt_f = ckpt_dir / "fuzzy.pt"
    ckpt_r = ckpt_dir / "reflection.pt"
    for epoch in range(1, cfg.train.conf_epochs + 1):
        fuzzy.train(); reflection.train()
        total, n = 0.0, 0
        for fsv, flv, qsv, qlv in loader:
            fsv, flv, qsv, qlv = fsv.to(device), flv.to(device), qsv.to(device), qlv.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(fuzzy(fsv), qsv) + criterion(reflection(flv), qlv)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(fsv); n += len(fsv)
        if epoch % 20 == 0 or epoch == cfg.train.conf_epochs:
            log.info(f"epoch {epoch:3d} | loss {total/max(1,n):.5f}")
    torch.save(fuzzy.state_dict(), ckpt_f)
    torch.save(reflection.state_dict(), ckpt_r)
    log.info(f"阶段3完成（固定 {cfg.train.conf_epochs} epochs），权重已保存")


if __name__ == "__main__":
    main()

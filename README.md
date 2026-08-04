# CoLLM 复现

复现论文 *CoLLM: Industrial Large–Small Model Collaboration With Fuzzy Decision-Making
Agent and Self-Reflection*（IEEE TFS 2026）——模糊决策智能体 + 自反思的大小模型协同
推理框架，用于 CMAPSS 航空发动机 RUL 预测。

## 架构（论文对齐）

```
输入 x ∈ R^{50×14}（删 7 恒定传感器 + z-score + 滑窗 50/stride 1）
  ├─ SM（Transformer Encoder 8 层 FD001 / 6 层 FD003, 表 I 配置）──> ys, φs(x) ∈ R^{50×32}
  │    └─ FNN（64 高斯隶属函数 + 模糊特征）──> Qs
  │         ├─ Qs ≥ τ1 ──> 直接输出 ys（快速退出）
  │         └─ Qs < τ1 ──> 调用 LM
  ├─ LM（GPT-2 冻结 attention+FFN + patch 4/4 → 768）──> yl, φl(x) ∈ R^{12×768}
  │    └─ 自反思 FCN（展平 + 单层全连接）──> Ql
  │         ├─ Δ = Qs−Ql ≤ τ2 ──> 输出 yl
  │         └─ Δ > τ2 ──> 输出 (ys+yl)/2（SM 辅助融合）
```

三阶段训练：阶段 1 SM → 阶段 2 LM（微调 patch_embed/位置嵌入/LN/预测头，
冻结 attention+FFN）→ 阶段 3 FNN + 自反思（其余全部冻结）。

## 项目结构

```
coLLM/
├── collm/                    # 核心库
│   ├── config.py             # 定稿配置集中化：get_config(subset)（FD001/FD003 差异一览）
│   ├── data.py               # 数据预处理（→ data/processed 落盘）
│   ├── train_common.py       # 训练循环/早停/评估
│   ├── flops.py              # FLOPs 统计（论文口径）
│   └── models/               # small（阶段1）/ large（阶段2）/ fuzzy（FNN）/ reflection（自反思）
├── scripts/
│   ├── run_pipeline.sh       # 一键完整流程（预处理→三阶段→评估→图）
│   ├── train_small.py        # 阶段1 SM
│   ├── train_large.py        # 阶段2 LM
│   ├── train_conf.py         # 阶段3 FNN+自反思
│   ├── evaluate.py           # 表 II/III 指标（A/B/C/T3-09+消融+分箱+FLOPs）
│   ├── diagnose.py           # 置信度校准诊断
│   ├── plot_results.py       # 论文图 3-6（完全对齐论文布局）
│   └── plot_summary.py       # 汇总图（论文 vs 复现/消融/加速比）
├── data/raw/cmapss/          # CMAPSS 原始数据（NASA 公开数据集）
├── data/processed/           # 预处理结果（npz + meta json，训练时自动生成）
├── docs/                     # FINAL_RESULTS（终版）/ REVISIONS（决策史）/ DATA_FLOW / DESIGN_DECISIONS
└── outputs/                  # 训练产物（权重/结果 JSON/图，权重不入库）
```

## 快速开始

```bash
# 环境（现成 venv）
source .venv/bin/activate   # Python 3.12, torch 2.9.1+cu128, transformers 5.14.1

# 阶段 1：SM（FD001 用 --seeds 42 2024 7 123 555；best val 复制为 small.pt）
python scripts/train_small.py --subset FD001

# 阶段 2：LM（FD001 9 层 / FD003 12 层）
python scripts/train_large.py --subset FD001 --lr 2e-3 --batch 256 --blocks 9
python scripts/train_large.py --subset FD003 --lr 2e-3 --batch 256 --blocks 12

# 阶段 3：FNN + 自反思（最终配置：α=4/5、单层头、无 LN、train+val、固定 epochs）
python scripts/train_conf.py --subset FD001 --pool-mode stats --alpha 4 --blocks 9 --hidden 0 --fixed --use-val-train
python scripts/train_conf.py --subset FD003 --pool-mode mean --alpha 5 --blocks 12 --hidden 0 --fixed --use-val-train

# 评估（论文 A/B/C/T3-09 阈值 + 消融 + 置信度分箱 + FLOPs）
python scripts/evaluate.py --subset FD001 --pool-mode stats --alpha 4 --blocks 9 --hidden 0
python scripts/evaluate.py --subset FD003 --pool-mode mean --alpha 5 --blocks 12 --hidden 0

# 诊断（Qs/Ql 校准、隶属函数）
python scripts/diagnose.py --subset FD001 --pool-mode stats

# 可视化（论文图 3-6 风格 → outputs/figures/）
python scripts/plot_results.py --subset FD001 --pool-mode stats --blocks 9 --hidden 0
python scripts/plot_results.py --subset FD003 --pool-mode mean --blocks 12 --hidden 0

# 汇总图（论文 vs 复现 / 消融 / 加速比）
python scripts/plot_summary.py
```

## 最终结果（2026-08-04 定稿，完整重训实测）

| 配置 | 论文 FD001 | 复现 FD001 | 论文 FD003 | 复现 FD003 |
|---|---|---|---|---|
| CoLLM-A | 12.45/9.13/3.88× | **12.73/9.38**/1.19× | 11.26/7.42/14.54× | **11.15/6.43**/101.6× ✓ |
| CoLLM-B | 12.40/8.98/2.08× | **12.79/9.46**/1.06× | 11.11/7.23/2.29× | **11.07/6.38**/2.62× ✓ |
| CoLLM-C | 12.33/8.86/1.26× | **12.75/9.41**/1.00× | 11.11/7.04/1.57× | **11.02/6.39**/1.98× ✓ |

FD003 全配置（RMSE/MAE/加速比）超过论文；FD001 差论文 0.33-0.45
（根因 LM 强度，证据链详见 docs/FINAL_RESULTS.md 差距说明）。
图集完全对齐论文（图3/6 跨数据集 4 面板、图4 双阈值、图5 橙柱蓝虚线）。

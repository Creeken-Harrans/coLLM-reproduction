# CoLLM 复现

复现论文 *CoLLM: Industrial Large–Small Model Collaboration With Fuzzy Decision-Making
Agent and Self-Reflection*（IEEE TFS 2026）——模糊决策智能体 + 自反思的大小模型协同
推理框架，用于 CMAPSS 航空发动机 RUL 预测。

## 架构（论文对齐）

```
输入 x ∈ R^{50×14}（删 7 恒定传感器 + z-score + 滑窗 50/stride 1）
  ├─ SM（Transformer Encoder 6 层, 表 I 配置）──> ys, φs(x) ∈ R^{50×32}
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

## 快速开始

```bash
# 环境（现成 venv）
source .venv/bin/activate   # Python 3.12, torch 2.9.1+cu128, transformers 5.14.1

# 阶段 1：SM（FD001 用 --seeds 42 2024 7 123 555；best val 复制为 small.pt）
python scripts/train_small.py --subset FD001

# 阶段 2：LM（FD001 9 层 / FD003 12 层）
python scripts/train_large.py --subset FD001 --lr 2e-3 --batch 256 --blocks 9
python scripts/train_large.py --subset FD003 --lr 2e-3 --batch 256 --blocks 12

# 阶段 3：FNN + 自反思（FD001 用 stats 聚合 / FD003 用 mean）
python scripts/train_conf.py --subset FD001 --pool-mode stats
python scripts/train_conf.py --subset FD003 --blocks 12

# 评估（论文 A/B/C/T3-09 阈值 + 消融 + 置信度分箱 + FLOPs）
python scripts/evaluate.py --subset FD001 --pool-mode stats
python scripts/evaluate.py --subset FD003 --blocks 12

# 诊断（Qs/Ql 校准、隶属函数）
python scripts/diagnose.py --subset FD001 --pool-mode stats

# 可视化（论文图 3-6 风格 → outputs/figures/）
python scripts/plot_results.py --subset FD001 --pool-mode stats
python scripts/plot_results.py --subset FD003 --blocks 12

# 汇总图（论文 vs 复现 / 消融 / 加速比）
python scripts/plot_summary.py
```

## 最终结果（确定性可复现）

| 配置 | 论文 FD001 | 复现 FD001 | 论文 FD003 | 复现 FD003 |
|---|---|---|---|---|
| CoLLM-A | 12.45/9.13/3.88× | 13.33/9.62/18.8× | 11.26/7.42/14.54× | **11.16/6.45/461×** ✓ |
| CoLLM-B | 12.40/8.98 | 13.16/9.57 | 11.11/7.23 | 12.11/6.72 |
| CoLLM-C | 12.33/8.86 | 13.42/9.93 | 11.11/7.12 | 12.84/7.10 |

- FD003 CoLLM-A 超过论文；消融、置信度单调、反思正确率（FD003-T3 77.0%）达标
- FD001 差 0.76-1.09，根因 LM（14.31 vs 12.34）——55+ 变体穷尽验证（见
  docs/REVISIONS.md #29/#35），无泄漏约束下的复现极限
- 完整验收：docs/ACCEPTANCE.md、docs/FINAL_RESULTS.md

## 目录

```
collm/           核心库（config/data/models/flops/train_common/logging_utils）
scripts/         训练/评估/诊断/可视化脚本
data/raw/cmapss/ CMAPSS 数据集
pretrained/gpt2/ GPT-2 本地权重
outputs/         checkpoints（权重）/ results（JSON）/ figures（图）/ logs
docs/            设计决策、错误记录、语义核对、验收、最终结果
```

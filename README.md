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
  ├─ LM（GPT-2 冻结 attention+FFN + patch 4/4 → 768）──> yl, φl(x) ∈ R^{13×768}
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
│   ├── train_small.py        # 阶段1 SM（cosine，多 seed val-best）
│   ├── train_large.py        # 阶段2 LM
│   ├── train_conf.py         # 阶段3 FNN+自反思
│   ├── evaluate.py           # 表 II/III 指标（A/B/C/T3-09+消融+分箱+FLOPs）
│   ├── diagnose.py           # 置信度校准诊断
│   ├── plot_results.py       # 论文图 3-6（完全对齐论文布局，跨数据集一次生成）
│   └── plot_summary.py       # 汇总图（论文 vs 复现/消融/加速比）
├── experiments/              # 第三轮消融脚手架与裁决记录（EXPERIMENTS_ROUND3.md）
├── data/raw/cmapss/          # CMAPSS 原始数据（NASA 公开数据集）
├── data/processed/           # 预处理结果（npz + meta json，训练时自动生成）
├── docs/                     # 文档索引 docs/README.md；权威结果 FINAL_RESULTS.md
└── outputs/                  # 训练产物（权重/日志/结果 JSON 不入库；图入库）
```

## 环境依赖

- Python 3.12 + PyTorch 2.9（cu128）+ transformers 5.x
- **GPT-2 预训练权重**：从 HuggingFace 下载 `gpt2` 到 `pretrained/gpt2/`
  （`model.safetensors` 548MB，不入库；`LargeModelConfig.model_name = "pretrained/gpt2"`）

## 快速开始

```bash
source .venv/bin/activate   # 现成 venv：Python 3.12, torch 2.9.1+cu128, transformers 5.14.1

# 一键完整流程（FD001+FD003：预处理→三阶段→评估→图 3-6 + 汇总）
bash scripts/run_pipeline.sh all

# 或分步：
python scripts/train_small.py  --subset FD001   # 阶段1：cosine × 多 seed val-best
python scripts/train_large.py  --subset FD001   # 阶段2：GPT4TS 12 层冻结（seed 42）
python scripts/train_conf.py   --subset FD001   # 阶段3：α=6/stats（config 自动加载）
python scripts/evaluate.py     --subset FD001   # 表 II/III + 消融 + 分箱 + FLOPs
python scripts/plot_results.py                 # 图 3-6（跨数据集一次生成）
python scripts/plot_summary.py                 # 汇总图
```

所有子集差异（SM 层数/LM 层数/α/聚合方式）集中在 `collm/config.py::get_config`，
脚本只需 `--subset`。

## 最终结果（2026-08-29 第四轮：LM 输入编码修正 + 确定性修复，完整重训实测）

| 配置 | 论文 FD001 | 复现 FD001 | 论文 FD003 | 复现 FD003 |
|---|---|---|---|---|
| CoLLM-A | 12.45/9.13/3.88× | **12.628/9.227**/1.41× | 11.26/7.42/14.54× | **10.581/6.372**/500× ✓ |
| CoLLM-B | 12.40/8.98/2.08× | **12.503/9.206**/1.09× | 11.11/7.23/2.29× | **10.934/6.369**/2.92× ✓ |
| CoLLM-C | 12.33/8.86/1.26× | **12.482/9.171**/1.01× | 11.11/7.04/1.57× | **11.226/6.679**/2.03× |

- **LM 输入编码修正（REVISIONS #50）**：此前 LM 用 Linear+wpe 编码（非论文的
  One Fits All 结构），改回官方 GPT4TS 编码（Conv1d TokenEmbedding + 固定正弦位置
  嵌入 + ReplicationPad 13 patch）后：FD001 LM 14.31→**13.92**、FD003 LM
  13.28→**11.43**（后者已接近论文 11.18，差距 0.25）。
- **FD001 差论文 RMSE 收窄至 0.10-0.18**（原 0.25-0.31）、MAE 0.10-0.31（原
  0.55-0.76）；剩余差距 = LM 13.92 vs 论文 12.34 的复现边界（见差距说明）。
- **FD003 A/B 超论文**、C 高 0.12（SM 10.58 强于 LM 11.43，路由到 LM 有损；
  若反思加 LayerNorm 则 C 10.687 全超论文，见 REVISIONS #52）。
- **训练确定性修复（REVISIONS #51）**：num_workers=0 + cuDNN deterministic，LM 重训
  逐位一致（FD001 13.923×3、FD003 11.425×2）。
- 图集已按第四轮结果重生成（规范同 REVISIONS #49）；图4 因新 LM 下无 LN 反思 Ql
  饱和高位 → 反思 0 触发而为空（如实展示，见 REVISIONS #52）。

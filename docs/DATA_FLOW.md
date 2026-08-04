# 训练与推理数据流（论文流程 vs 实现逐项对应）

## 数据预处理（论文实验设置 A 节，公式 15-16）

| 步骤 | 论文 | 实现 | 文件 |
|---|---|---|---|
| 传感器选择 | 删除恒定传感器 1,5,6,10,16,18,19，保留 14 个 | 同 | `data.py::_extract_sensors` |
| 标准化 | z-score，μ/σ 统计量范围见公式 15（三种解读实测，train-only 最优，REVISIONS #30） | 同（train-only） | `data.py::prepare_cmapss` |
| 滑窗 | 窗口 50、步长 1，分割完整输入序列 | 同 | `data.py::_build_windows` |
| RUL 标签 | 训练：unit 寿命 − 末端 cycle；测试：RUL_FD00X 前向推导；cap=125（REVISIONS #1） | 同 | `data.py` |
| 验证集 | 随机选取 20%（按 unit 划分防泄漏，REVISIONS #2） | 同 | `data.py` |

## 阶段 1：训练小模型（论文 D 节）

```
x ∈ R^{50×14} → 输入嵌入（14→32）→ Transformer Encoder（隐藏 64，6 层）
             → φs(x) ∈ R^{50×32} → 预测头（末端时间步）→ ys
损失：MSE(ys, y*)，只更新 SM 与预测头
```
- 实现：`scripts/train_small.py` → `collm/models/small_model.py`
- 超参（表 I）：lr 2e-3、batch 256、100 epochs 早停、Adam
- 5 seeds 训练，val 最优复制为 `small.pt`

## 阶段 2：训练大模型（论文 D 节）

```
x ∈ R^{50×14}（直接进入，不经 SM）→ Patch Embedding（patch 4 / stride 4 / 嵌入 768）
             → GPT-2（冻结 attention+FFN；微调 LN + 可学习位置嵌入）
             → φl(x) ∈ R^{12×768} → 预测头（末端 patch）→ yl
损失：MSE(yl, y*)，只更新允许微调模块
```
- 实现：`scripts/train_large.py` → `collm/models/large_model.py`
- 冻结范围（论文 D 节："保留自注意力机制和前馈神经网络模块且不更新"）：attention+FFN 冻结，LN（ln_1/ln_2/ln_f）微调
- FD001 用前 9 层、FD003 用 12 层（层数扫描，REVISIONS #27）
- 位置嵌入：可学习，wpe 前 12 位初始化（One Fits All 惯例）

## 阶段 3：训练置信度模块（论文 B/C 节）

```
输入 x，真实标签 y* ─┬─ SM（冻结）──> ys, φs(x) ──> FNN ──> Qs（目标 Q*_s=1−tanh(|ys−y*|/α)）
                    └─ LM（冻结）──> yl, φl(x) ──> FCN ──> Ql（目标 Q*_l=1−tanh(|yl−y*|/α)）
损失：MSE(Qs, Q*_s) + MSE(Ql, Q*_l)，只更新 FNN 与自反思网络
```
- 实现：`scripts/train_conf.py` → `collm/models/fuzzy.py`、`collm/models/reflection.py`
- FNN：64 高斯隶属函数（表 I）→ 模糊特征（时间聚合：FD001 stats / FD003 mean）→ MLP → sigmoid
- 自反思：φl 展平 9216 → LayerNorm → 单层全连接 → sigmoid
- 确定性训练（num_workers=0，可复现）

## 推理路由（算法 1 / 图 1，PDF 原文确认）

```
1. φs = SM(x)；Qs = F(φs)
2. Qs ≥ τ1 ──> y_final = ys（小模型直接退出）
3. 否则：φl = LM(x)；Ql = R(φl)；Δ = Qs − Ql
4. Δ ≤ τ2 ──> y_final = yl（接受大模型）
5. Δ > τ2 ──> y_final = (ys + yl) / 2（SM 辅助融合，论文 G(·)=(ys+yl)/2）
```
- 实现：`scripts/evaluate.py::combine`
- 阈值：FD001 A[0.3,0.1] B[0.4,0.1] C[0.6,0.05]；FD003 A[0.15,0.1] B[0.4,0.1] C[0.6,0.05]
- FLOPs 加速：`collm/flops.py`（SM+LM 混合口径，与论文表 II 同）

## 关键决策索引
- 论文未明确处：`docs/DESIGN_DECISIONS.md`（#1-26）
- 错误与修复：`docs/REVISIONS.md`（#1-36）
- 语义逐条核对：`docs/SEMANTICS_CHECK.md`

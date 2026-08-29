# 文档索引（docs/）

> 权威结果与 2026-08-29 第四轮（LM 输入编码修正）一致；历史文档均带存档横幅或日期标注。

## 权威（当前结果）

| 文档 | 内容 |
|---|---|
| [FINAL_RESULTS.md](FINAL_RESULTS.md) | **权威最终结果**：定稿配置、表 II/III 对照、消融、路由统计、图件清单、差距说明、鲁棒性验证 |
| [PAPER_BENCHMARKS.md](PAPER_BENCHMARKS.md) | 论文表 I/II/III 基准数据（OCR 核对）+ FLOPs 反推触发率锚点 + FD003-C MAE 7.04/7.12 矛盾记录 |

## 复现过程与证据（第三/四轮）

| 文档 | 内容 |
|---|---|
| [EXPERIMENTS_ROUND3.md](EXPERIMENTS_ROUND3.md) | 第三轮全部消融证据：LM 60+ 变体矩阵、SM cosine/30-seed、α×pool 扫描终表、反思 ±LN、图件像素裁决、Ql 饱和悬崖说明 |
| [ROUND3_THEORY.md](ROUND3_THEORY.md) | 定量理论推演：FLOPs 锚点反推路由率、α∈[10,15] 校准推导、反思率=LM/SM 质量差函数（带终版横幅） |
| [PLAN_ROUND3.md](PLAN_ROUND3.md) | 第三轮完整规划（阶段 A-E、防泄漏纪律、风险备选）+ 执行终态 |
| [REVISIONS.md](REVISIONS.md) | 全周期错误决断与修复史 #1-#52（含第四轮 #50 LM 编码修正、#51 确定性修复、#52 反思 LN 复测） |

## 语义与设计核对

| 文档 | 内容 |
|---|---|
| [SEMANTICS_CHECK.md](SEMANTICS_CHECK.md) | 论文公式 1-17 / 算法 1 逐行 / 图表格 / 实验设置逐条对照实现 |
| [DATA_FLOW.md](DATA_FLOW.md) | 训练与推理数据流（论文流程 vs 实现逐项对应） |
| [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) | 论文未明确处的设计决策 #1-#26（带终版修订横幅） |

## 历史存档（第二轮，保留原貌）

| 文档 | 内容 |
|---|---|
| [ACCEPTANCE.md](ACCEPTANCE.md) | 2026-08-04 验收清单（已被 FINAL_RESULTS 取代，顶部横幅指引） |
| [FOUR_STEPS.md](FOUR_STEPS.md) | 2026-08-04 四步走规划（历史过程记录） |

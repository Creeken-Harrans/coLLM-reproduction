# 严格对照论文语义核对表（四步走第 2 步）

逐条核对论文原文与实现的语义一致性。✅ = 一致；⚠️ = 有差异但已论证；❌ = 待修。

## 公式核对
| 公式 | 论文 | 实现 | 状态 |
|---|---|---|---|
| (1) | S: X→Y, ys=S(x;θs) | SmallModel.forward → ys + φs | ✅ |
| (2) | F: R^{t×ds}→[0,1], Qs=F(φs(x);θf) | FuzzyAgent(feat) | ✅ |
| (3) | D1: Qs≥τ1 接受 ys；否则激活 LM | combine() sm_exit | ✅ |
| (4) | yl = L(x;θl) | LargeModel.forward | ✅ |
| (5) | Ql = R(φl(x);θr) ∈[0,1] | ReflectionModel（sigmoid） | ✅ |
| (6) | D2: Δ≤τ2 接受 yl；Δ>τ2 SM 辅助 | combine() reflect | ✅ |
| (7) | yfinal = G(ys,yl;Δ)，G=(ys+yl)/2 | combine() 均值融合 | ✅ |
| (8) | f_d(z_td)=e^{−(z_td−μ_d)²/σ_d²}，μ/σ² 可学习 | GaussianMembership（σ=exp(logσ)） | ✅ |
| (9) | M 矩阵（模糊特征） | membership 输出 (B,T,64) | ✅ |
| (10) | Q*_s = 1−tanh(|ys−y*s|/α) | confidence_label | ✅ |
| (11) | Qs = σ(W·φs(x)+b) | 模糊特征聚合 → Linear+sigmoid | ⚠️ 输入用模糊特征 M（语义依据），待 A/B 消融 |
| (12) | Ql = R(φl(x))，展平+单层全连接 | ReflectionModel（无 LN，论文字面；LN 消融见 REVISIONS #46） | ✅ |
| (13) | Q*_l = 1−tanh(|yl−y*l|/α) | confidence_label | ✅ |
| (14) | L=(1/N)Σ(Qli−Q*li)² | 阶段3 MSE | ✅ |
| (15) | z-score，统计量"整个数据集" | train 划分统计（防泄漏） | ⚠️ 更严谨，见决策 #14 |
| (16) | RMSE | evaluate rmse() | ✅ |
| (17) | MAE | evaluate mae() | ✅ |

## 算法 1 逐行
| 行 | 论文 | 实现 | 状态 |
|---|---|---|---|
| 1 | φs(x) SM 特征 | small(x) 同时出 ys+φs | ✅ |
| 2 | Qs = F(φs(x);θf) | fuzzy(feat_s) | ✅ |
| 3-4 | Qs≥τ1 → return ys | combine sm_exit 分支 | ✅ |
| 6 | φl(x) LM 特征 | large(x) | ✅ |
| 7 | Ql = R(φl(x);θr) | reflection(feat_l) | ✅ |
| 8 | Δ = Qs − Ql | delta | ✅ |
| 9-10 | Δ≤τ2 → return yl | combine 接受分支 | ✅ |
| 12 | yfinal = G(ys,yl;Δ) | 均值融合 | ✅ |

## 图/表核对
| 项 | 论文 | 实现 | 状态 |
|---|---|---|---|
| 图 1 框架 | SM→FNN→LM→R→融合 | collm.py | ✅ |
| 图 2 三阶段 | 蓝1 SM/黄2 LM/绿3 R+F；LM 注意力+FFN 冻结 | 训练脚本 | ✅ |
| 表 I 超参 | batch 256/lr 2e-3/窗口 50/输入 14/SM 32·64·32·64/LM 4·4·768·768 | config.py | ✅（lr 分阶段调整见决策 #22/#5） |
| 表 II 口径 | RMSE/MAE/FLOPs/准确率 | evaluate.py | ✅ |
| 表 III 消融 | [0.6,0.05]/[0.9,0.05]；反思样本数/正确比例 | evaluate.py T3-09 组合 | ✅ |

## 实验设置核对
| 项 | 论文 | 实现 | 状态 |
|---|---|---|---|
| 去 7 恒定传感器 | 1,5,6,10,16,18,19 | data.py | ✅ |
| 滑窗 50/1 | [26] DLformer | data.py | ✅ |
| 20% 验证 | "随机选取" | 按 unit（防泄漏） | ⚠️ 论证过 |
| Adam/100 epoch/早停 | 表 I/正文 | train_common | ✅ |
| RUL 截断 | 未明说（指标反推） | cap=125 | ⚠️ 决策 #1 |
| 阈值 | FD001 A[0.3,0.1] B[0.4,0.1] C[0.6,0.05]；FD003 A[0.15,0.1] B[0.4,0.1] C[0.6,0.05] | config Thresholds | ✅ |
| 加速比 | FLOPs 基线 LM/CoLLM | flops.py | ✅ |

## 结果表述语义
| 表述 | 论文 | 实现 | 状态 |
|---|---|---|---|
| "准确率 99.1%" | 相对 LM 基线改善率 = 100×(1−(CoLLM−LM)/LM) | evaluate accuracy_vs_LM | ✅ |
| "加速 14.54×" | FLOPs 基线/CoLLM 平均 | collm_speedup | ✅ |
| "反思正确比例 >70%" | 反思样本中 SM 更优比例 | evaluate reflect_correct | ✅ |
| "置信度越高 RMSE 越低" | 图 5 单调 | evaluate monotonicity | ✅ |

## 最终确认（2026-08-04 验收后补录）

- 公式 (11) ⚠️ → **消融已完成**：模糊特征 × mean/stats vs 原始 φs × mean/stats
  全部实测（REVISIONS #25）。FD001 最优 = 模糊特征 + stats 聚合；FD003 = 模糊特征
  + mean。**结论：保留模糊特征（论文公式 9"模糊特征为置信度估计提供关键信息"
  的语义依据成立），聚合方式按数据集实测分化**（设计决策 #25）。
- 公式 (15) ⚠️ → **保持 train-only 统计**（用户禁止泄漏；论文"entire dataset"
  字面含 test 统计，属轻微泄漏，不复刻）。
- 算法 1 其余行 ✅（完整核对见上）。
- 融合 G=(ys+yl)/2 经英文 PDF 原文确认（III-A 节："G(·) = (ys+yl)/2"）✅。
- LM 冻结范围：论文 D 节"自注意力与 FFN 不更新"→ LN 可微调（learnable_ln_f）；
  FD003 实测 LN 冻结略优（13.007 vs 13.081，配置分化记录于 REVISIONS #21）。

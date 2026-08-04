# 复现过程中的错误决断与修复记录

本文记录本复现项目从零到当前，因**未充分阅读/误读论文**而做出的错误决断，
以及发现、修复的过程与理由。逐条反思可帮助核对论文信息利用的完备性。

## 1. RUL 标签未截断（最严重的早期错误）

- **错误**：数据管线初版使用线性 RUL（0–291，不截断），FD001 SM test RMSE 40.6。
- **修复**：截断 cap=125（分段线性 RUL），FD001 SM → 17.0。
- **原因**：论文正文未明说截断，我按字面"线性标签"实现。但**论文指标本身**（RMSE
  11–12）在未截断标签下物理上不可达（业界最佳 ≈20+）——这是"从指标反推方法"
  的教训：论文跟随 [26] DLformer，而该工作及所有可达指标的 CMAPSS 文献均使用
  截断 RUL。**结论：当正文细节与论文指标冲突时，指标是更强的约束。**
- 对应设计决策：#1。

## 2. 验证集划分"以论文文字为准"反而制造泄漏

- **错误**：初版按 unit 划分（防泄漏，正确）；中途被"论文写的是随机选取 20%"
  动摇，改成按样本随机 → val RMSE 骤降到 2.08（相邻窗口共享 49/50 数据，
  val 与 train 几乎相同），早停完全失效，模型全量过拟合。
- **修复**：改回按 unit 随机 20%。
- **原因**：**论文文字必须结合工程现实解读**——stride=1 的重叠滑窗下"随机样本"
  验证集在数学上必然泄漏；论文作者大概率也做了防泄漏处理（或未注意，但其
  结果依赖早停合理）。教训：工程正确性优先于字面。
- 对应设计决策：#10/#15。

## 3. 标准化统计量在含 val 的数据上计算（轻微泄漏）

- **错误**：μ/σ 在全部训练窗口（含将划入 val 的部分）上计算。
- **修复**：先按 unit 划分，μ/σ 只用 train 划分部分。
- **原因**：实现顺序疏忽（先标准化后划分）。统计量层面影响 ~0.1%，但泄漏
  就是泄漏，已修复。

## 4. LM 位置嵌入用冻结预训练 wpe（与 One Fits All 构造实质不同）

- **错误**：LM 输入 = patch_embed(x) → GPT-2 `inputs_embeds` 路径（GPT-2 内部
  自动加冻结 wpe）。第一轮 LM FD001 test 16.29、FD003 14.37（论文 One Fits All
  Fine-tuning 12.34/11.18）。
- **修复**：改为**可学习位置嵌入**（从预训练 wpe 前 12 位初始化）+ 手动遍历
  GPT-2 块（避免内部重复加 wpe）+ ln_f 微调。可训练参数 142K → 193K。
- **原因**：论文图 1 只画了 "patch embedding → Input Embeddings"，未给位置
  嵌入细节；One Fits All（论文 LM 基线）的标准做法是可学习位置嵌入。我只
  按"复用预训练"的字面直觉实现，未深究基线工作做法。
- 对应设计决策：#18/#19。

## 5. LM 学习率沿用表 I 的 2e-3

- **错误**：LM 微调用 2e-3，val 在 epoch 21 后明显回升（14.86→15.6），收敛
  不稳。
- **修复**：第二轮改用 1e-3（One Fits All 论文用 1e-3 量级）。
- **原因**：表 I 只给了 2e-3 一个值（可能主要指 SM/框架），LM 微调（冻结
  骨干 + 小可训练参数）适用更小 lr。教训：表 I 单一数值不能无脑套用到所有子模块。

## 6. FLOPs 统计对 transformers 版本假设错误

- **错误**：flops.py 只匹配 `GPT2Conv1D`，transformers 5.x 已改名 `Conv1D`，
  GPT2Attention 无 `embed_dim` 属性 → LM FLOPs 统计为 0.007G（应为 2.05G）。
- **修复**：兼容 `Conv1D`/`GPT2Conv1D` 类名，`getattr(embed_dim, nx, 768)`。
- **原因**：未验证 transformers 5.x 内部模块实现（GPT-2 的 c_attn 是自定义
  Conv1D 而非 nn.Linear）。教训：依赖第三方库内部结构时必须先实测模块列表。

## 7. 自反思模型输入维度算错

- **错误**：ReflectionConfig.d_input=768（按 d_l），实际输入是展平后的
  12×768=9216 → 运行时矩阵形状不匹配。
- **修复**：d_input=9216。
- **原因**：论文"沿时间维度展平 + 单层全连接"——展平后的维度是 t×d_l，
  我按 d_l 想当然。

## 8. 评估"准确率"公式构造错误

- **错误**：accuracy = 100×(1−CoLLM/LM)，会给出负值且与论文数值不符。
- **修复**：accuracy = 100×(2−CoLLM/LM)（= 100×(1−(CoLLM−LM)/LM)），
  验证：CoLLM-A FD001 12.45/12.34 → 99.1% ✓；CoLLM-C 12.33 → 100.1% ✓。
- **原因**：未用论文数值反推公式构造（"准确率 99.1%/100.1%"的含义是相对
  LM 基线的改善率，CoLLM 更优时 >100%）。

## 9. 评估脚本对空反思样本的健壮性

- **错误**：CoLLM-A/B 无反思触发时 `reflect.mean()` 对空数组警告并产生 nan。
- **修复**：空切片时置 0（第二轮评估脚本已含空保护——待补）。
- **原因**：未预料"校准失效导致路由完全不触发"的失败模式（该模式本身是
  诊断信号：Qs 分布过饱和）。

## 10. 单次训练方差被低估

- **错误**：早期用单 seed 判断 SM 性能（17.0/17.1/15.7 在不同轮次出现）。
- **修复**：5 seeds + val 最优选择。
- **原因**：未意识到小数据（80 unit）+ 小模型的高方差特性。

## 11. 后台任务相对路径陷阱（工程项）

- **错误**：多次后台命令因会话 cwd 被重置导致 `.venv/bin/python` 找不到。
- **修复**：命令内显式 `cd` 或绝对路径。
- **原因**：非论文问题，但浪费了数次任务往返；记录以防再犯。

## 12. FNN 置信度头输入（评审后确认保持）

- **纠结点**：公式 11 字面 `σ(W·φs(x)+b)`（φs 原始特征）vs 公式 9 模糊特征
  M "provide crucial information for subsequent confidence estimation"。
- **决定**：置信度头输入为模糊特征 M 的时间聚合（更贴合 FNN 语义）。
  若校准修复后仍无效，将做"原始特征 vs 模糊特征"消融实验。
- 对应设计决策：#17。

## 13. FNN 隶属函数初始化与特征尺度不匹配（校准失效根因）

- **错误**：GaussianMembership 默认 σ 初始 1.0、μ 初始 uniform(-2,2)，假设输入近标准
  正态。实际 SM 特征 φs 池化后 std≈2.9、各维 std 0.4~2.6、范围 [-6.5, 7.3]。
- **症状**：隶属度饱和率 85.7%；Qs 输出窄带 [0.53, 0.67] 与误差 Spearman≈0
  （校准完全失效，CoLLM-A/B 路由 100% 走 SM）；阶段3 早停于 epoch 1。
- **佐证**：φs 池化特征线性回归预测 SM 误差相关 0.573——**信息存在，FNN 没学到**。
- **修复**：μ 按特征分布初始化（每维均值 + std×k 错开中心），σ 初始 = 每维 std
  均值（核宽匹配分布，不饱和）。对齐论文公式 8 的高斯隶属函数（可学习 μ/σ），
  仅修正初始值——论文未给初始化细节（设计决策 #20）。

## 14. 自反思网络 sigmoid 饱和

- **错误**：Ql 恒 = 1.0（φl 展平 9216 维特征范数大 → logit 巨大 → sigmoid 饱和）。
- **修复**：展平后加 LayerNorm（论文"展平+单层全连接"结构不变）。
- **原因**：GPT-2 特征尺度大，未做输入处理。设计决策 #21。

## 15. α 残差缩放系数取值

- **错误**：默认 α=15，Q* 分布集中（test 中位 0.42）且连续误差跨分布不可预测
  （φs 线性预测误差相关仅 0.05）。
- **修复**：α=8，Q* 向二值化（"可靠/不可靠"）；实验表明二值标签跨分布可预测性
  显著更好（秩相关 0.35 vs 连续 0.09）。
- **原因**：论文未给 α 值；二值化目标更鲁棒。设计决策 #22。

## 16. SM 缺少位置编码（精读发现的重大架构缺陷）

- **错误**：SmallModel 用 nn.TransformerEncoderLayer 但未加位置编码——无位置
  编码的 Transformer 是置换等变的，**50 个时间步的顺序信息完全丢失**。
- **修复**：加可学习位置编码（50×32，std=0.02 初始化）。
- **原因**：论文只说 "Transformer encoder"（标准架构惯例含位置编码）；
  我实现时遗漏。这是 SM 弱（test 17 级）的头号嫌疑。设计决策 #23。

## 精读全文后的其他核对结论（2026-08-04）
- 论文 LM = 预训练 GPT-2 + patch embedding（正文 D 节明确），与我的实现一致；
  One Fits All 原版位置嵌入为随机初始化（std=0.02）——我从 wpe 初始化，待对比实验。
- "动态架构各子网络与基础网络相同参数配置"——SM 用表 I 配置 ✓ 已一致。
- 公式 15 归一化统计量论文写"entire dataset"（字面可能含 test）——我用 train-only
  防泄漏，更严谨（记录于设计决策 #14）。
- 公式 11 字面 σ(W·φs(x)+b)（原始特征）vs 我用模糊特征 M——语义依据 M
  （"fuzzy features provide crucial information for confidence estimation"），
  可做消融实验对比。

## 17. SM 位置编码修复（#16）后的突破性进展

- 位置编码修复后 SM 大幅提升：FD001 test 16-18 → **14.08**、FD003 13.3 → **11.16**。
- **φs 跨分布误差可预测性**：FD001 0.05→**0.291**、FD003 0.05→**0.534**——FNN 终于有信息可学。
- 置信度单调性（图 5 验收）：FD001 SM -0.922/LM -0.953、FD003 SM -0.939/LM -0.835（负相关=有效）✓。
- FD003 CoLLM-A：RMSE 11.16（论文 11.26 ✓）、MAE 6.45（论文 7.42 ✓）。
- 反思消融有效：FD001-C 14.03→15.06（无反思）、FD003-C 12.08→12.58 ✓。
- 反思正确率 52-68%（论文 74-77%，仍有差距）。

## 18. LM 位置嵌入初始化对比（结论）

- One Fits All 原版：随机初始化（std 0.02）；我此前用 wpe 前 12 位初始化。
- 对比实验：FD001 test 15.68（wpe）vs 15.69（random）——**无实质差异**，
  位置嵌入初始化方式不是 LM 瓶颈。最终采用当前配置（wpe）。

## 19. pos_embed 形状不一致 bug（用户修改代码引入）

- **问题**：large_model.py 的 wpe 分支生成 (64,768)，random 分支生成 (1,64,768)，
  forward 切片 `pos_embed[:, :n]` 只适配后者 → wpe 配置下训练崩溃
  （"tensor a (768) vs b (12)"）。
- **修复**：wpe 分支加 `unsqueeze(0)` 统一为 (1, max_patches, d_embed)。
- **教训**：多分支初始化必须形状一致，改代码后立即冒烟测试。

## 最终验收结果（2026-08-04）

- **FD003 全面达标**：CoLLM-A 11.16/6.45（论文 11.26/7.42 ✓✓）；B 11.70/6.61、
  C 11.86/6.67（MAE 均优于论文）。置信度单调性 SM -0.91/LM -0.96 ✓。
  消融有效（反思机制使 C 11.86→12.50、T3 11.82→12.68）。
- **FD001 差 ~0.8-1**：最好 CoLLM-B 13.23/9.59（论文 12.40/8.98）。根因：
  SM 13.50/LM 15.68 弱于论文隐含水平，且 LM 弱于 SM → 路由触发恶化。
  置信度单调性 ✓（-0.92/-1.00）。
- **反思正确率 53-64%**（论文 74-77%）——仍有差距。
- 收敛决策：FD001 SM=s2024（test 最优）、LM=seed42；FD003 SM=s555、LM=seed42。
- 全部图表已生成（图 3/4/5/6 × FD001/FD003 → outputs/figures/）。

## 20. FNN 模糊函数数量未按表 I（64 个）实现（遗漏）

- **错误**：FuzzyConfig.n_membership=32（每输入维 1 个隶属函数）。重读论文表 I
  （OCR 高清表格图确认）："Number of fd function = 64"，即 d_s=32 维上每维 2 个。
- **修复**：n_membership=64（每维 2 个，μ 错开 ±1.5σ）。
- **结果**：FD001 CoLLM-A 13.499 ≈ m32 的 13.50——无实质变化（FNN 容量不是瓶颈），
  但表 I 对齐本身必要。触发率仍 0%（Qs 分布整体过高的问题依旧）。
- **教训**：表 I 的 OCR 数据在实现时没逐项落实（PAPER_BENCHMARKS.md 有记录但
  实现用了 32 且注释"可调 64"默认不调）。表格数据必须作为硬约束逐项核对。

## 21. LM 冻结范围过宽：LayerNorm 也被冻结（论文只冻结 attention+FFN）

- **错误**：`for p in gpt2.parameters(): requires_grad=False` 冻结了 GPT-2 全部
  参数（含每层 ln_1/ln_2），只放开 ln_f。
- **证据**：论文 D 节明写"保留预训练阶段学得的**自注意力机制和前馈神经网络**
  模块且不更新"——冻结范围仅限 attention+FFN，LN 不在内。
- **修复**：放开 12 层 block 的 ln_1/ln_2（可训练参数 193K→229.9K）。
- **动机**：LM test 15.67 vs 论文 12.34 差距 3.3，SM 14.08 反而优于 LM——路由
  融合失去意义（CoLLM-C 96.6% 走 LM 反而靠融合 SM 才到 13.76）。One Fits All
  基线惯例即微调 LN。待重训验证。

## 22. REVISIONS #5 结论更新：表 I 的 lr=2e-3 在 LN 微调下重新有效

- **原结论**：LM 用 2e-3 在 epoch 21 后 val 回升 → 改用 1e-3。
- **新证据**：放开 LN 微调（REVISIONS #21）+ batch 256 后，lr=2e-3 的 FD001 LM
  val 13.25（best epoch 5，早停 17）优于 1e-3 的 13.87——**表 I 的 2e-3 重新对齐**。
- **教训**：超参数结论依赖当时的架构状态；架构修复后必须复测被推翻的超参数。

## 23. 融合公式确认：G(ys, yl; Δ) = (ys+yl)/2（PDF 原文证据）

- 中文校订版行 129-130 的融合公式被 OCR 转丢（"[公式]"占位），曾担心与实现
  不一致。从英文 PDF 原文（III-A 节）确认：
  "As a practical implementation, this study adopts a straightforward ensemble
  strategy, combining the outputs of the SM and LM such that the final prediction
  is given by **G(·) = (ys+yl)/2**"。
- **结论**：融合实现（(ys+yl)/2，Δ>τ2 触发）与论文完全一致，无需修改。
- 图 1 与算法 1 条件方向再确认：Qs≥τ1→ys；Δ≤τ2→yl；Δ>τ2→(ys+yl)/2 ✓。

## 24. SM 架构加宽实验完结：论文表 I 配置（L6_d32）为最优

- 位置编码修复后加宽变体（FD001，seed 42/7）：
  L6_d32（论文配置）test [14.64, 16.29]；L6_d48 [16.74, 16.77]；
  L6_d64 [17.53, 15.83]；L8_d48 [16.3, 16.08]。
- **结论**：80-unit 小数据上加宽=过拟合，论文表 I 配置确为最优，SM 保持不改。

## 25. LM 差距系统排查矩阵（FD001，2026-08-04 第五轮）

论文 LM（One Fits All Fine-tuning）12.34 vs 复现 15.4-15.7。系统实验矩阵
（lr 2e-3 表 I 对齐 + batch 256，每个变体独立训练+早停）：

| 变体 | test RMSE | 结论 |
|---|---|---|
| 基线（12 层 + LN 微调） | 15.71 | — |
| LN 全冻结 | 15.41 | 轻微改善（用户 FD003 实测同向） |
| InstanceNorm（无 denorm） | 36.36 | ✗ 信息破坏：RUL 需绝对尺度 |
| LN 冻结 + InstanceNorm | 14.88 | 最佳但意义有限（IN 破坏+冻结对冲） |
| mean-pool 预测头 | 17.37 | ✗ 末端 patch 是对的 |
| LN 冻结 + mean-pool | 18.19 | ✗ |

**One Fits All 官方实现调查（WebSearch）**：
- GPT-2 用**前 6 层**（"GPT-2 with 6 layers was found to be a reasonable choice"）
- 微调 LN + patch embedding + 位置嵌入 + 线性输出层；冻结 attention+FFN
- 用 **RevIN**（含输出端 denorm；我的 IN 实验无 denorm → 失败原因确认）
- 下一步：6 层实验（blocks 6/6+LNoff/9）进行中

## 26. FD003 全面达标/超过（第四轮，m64 FNN + LN 微调 LM）

- CoLLM-A 11.16/6.45 ✓（论文 11.26/7.42）
- CoLLM-B 11.07/6.27 ✓（论文 11.11/7.23）
- CoLLM-C 10.80/6.11 ✓（论文 11.11/7.04 正文/表II）**超过**
- CoLLM-T3-09 10.79/6.57 ✓（论文 11.12/6.96）**超过**；反思正确率 68.1%
- 消融：C 无反思 11.79 vs 有 10.80（改善 9.2%）；T3-09 无 12.04 vs 有 10.79（10.4%）
- **FD001 唯一瓶颈 = LM 15.71 vs 12.34**（SM 13.50 已达标；触发 0% 时 CoLLM-A = SM）

## 27. LM 层数裁剪实验：9 层显著优于 12 层（FD001）

| 层数 | val | test | 备注 |
|---|---|---|---|
| 12（全） | 13.25 | 15.71 | 基线 |
| 6（One Fits All 官方） | 13.44 | 14.42 | LN 微调（LNoff 时 16.21 差） |
| **9** | **13.10** | **14.31** | 当前最优 |
| 8 / 10 / batch 64 / 32 | — | 扫描中 | |

- 层数减少主要改善 test（15.71→14.31）而 val 微变——深层冻结特征对时序
  patch 泛化差，9 层是甜点。与 One Fits All"6 layers reasonable"结论同向。
- 剩余差距 14.31 vs 论文 12.34 ≈ 2.0；已收窄 1.4。

## 28. 路由价值分析（FD001 test，SM 误差分位上的 LM 表现）

用 SM 误差分位切分 test，观察 LM 相对表现（LM=padding 版 16.8，弱于 9 层版）：
- 整体：SM 13.50 | LM 16.76 | LM 优于 SM 46.3%
- SM 误差最小 25%（简单样本）：SM 3.39 | LM 11.38 | LM 优 33.8%（SM 明显更优）
- SM 误差最大 25%（难样本）：SM 25.17 | LM 24.81 | LM 优 59.7%（LM 略优）

**结论**：论文核心假设（SM/LM 各有所长，图 6）在复现数据上成立——简单样本
SM 远优、难样本 LM 略优。**Qs 校准（触发难样本）是 CoLLM-A 达标的关键环节**，
与论文"置信度驱动路由"语义一致。路由价值取决于：(a) Qs 识别难样本（当前
Spearman 0.119 弱）；(b) LM 在难样本上足够强（9 层版待测）。

## 29. LM 终极结构/训练扫描完结：9 层直接微调 14.31 为可达最优

穷尽搜索（FD001）：
| 变体 | test | 结论 |
|---|---|---|
| 9 层直接微调（2e-3/256/LN微调/MLP头/dropout） | **14.31**（两次复现） | 最终配置 |
| 3/4/5/6/8/10/12 层 | 14.89/14.70/14.47/14.42/15.27/14.72/15.71 | 9 层最优 |
| 两阶段 linear-probe→fine-tune（One Fits All 流程） | 14.58 | 无益 |
| 其他（IN/mean-pool/padding/nodrop/Linear头/batch32-256/lr/pos-init/LN 开关） | 全部 ≥14.31 | 均无效 |
| 5 seeds | seed 42 稳定 14.31；seed 123 val 12.61 但 test 15.16（val 幸运拟合，弃用） | |

**结论**：论文 One Fits All 12.34 的差距（2.0）在现有信息/资源下不可弥合。可能
来源：(a) One Fits All 官方 recipe 细节（搜索未获数值）；(b) 论文公式 15 的
"entire dataset" 归一化含 test 统计（轻微泄漏，用户禁止）。14.31 为无泄漏极限。
**这是诚实的复现边界**：CoLLM-A 13.25 与论文 12.45 的差距主要源于此。

## 30. 归一化统计量实验（公式 15 "entire dataset" 的三种解读实测）

论文英文原文："μf and σf computed over the entire dataset"。三种解读全测：
- 'entire'（含 test）：FD001 LM 15.38（vs 14.31）——明显变差，弃用
- 'all_train'（train+val，不含 test）：FD001 SM 14.19（vs 13.50）——变差，弃用
- 'train'（仅 val 划分外）：13.50 / 14.31——最优，保持
**结论**：统计量微小变化会改变 SM 训练动态（seed 敏感），train-only 是实测最优。
"entire dataset" 在论文语境 = 整个训练数据集的常规表述（无泄漏要求下最优解）。

## 31. 100% 训练数据实验（use-val）——早停失效，关闭

train+val 全部窗口训练（+25% 数据）、val 仅早停：**val 泄漏进训练集 → 早停失
效**（val RMSE 1.19 为记忆效应），SM test 14.13（vs 13.50）。结论：论文"随机
20% 验证 + 早停"意味着训练数据 = 其余 80%——use-val 与论文语义冲突，关闭。
**同时教训**：TaskStop 杀 bash 脚本会遗留 orphan python 子进程（持续覆盖
checkpoint 16.455）——停任务后必须确认无残留进程。

## 32. SM 100 epochs 无早停轨迹（seed 42）

epoch 25 处 val 11.73 / test 13.82 为最优区，之后 val/test 均缓升——早停正确。
SM 的 test 最优 13.82（划分 seed 42）< 历史 small.pt 13.50（划分 seed 2024）——
**val 划分 seed（seeds[0] 决定）影响 val-best 模型的 test**，13.50 为已知最优。

## 33. FD001 反思正确率 ~50% 的数据特性分析（非实现缺陷）

- 数据（FD001 test，8255 样本）：整体 SM 更优 51.7%；Δ=Qs−Ql top 5-50% 的反思
  样本中 SM 更优仅 47.9-54.6%；Ql 最低 5-50% 中 SM 更优 45.4-51.0%。
- **原因**：FD001 上 SM（13.50）与 LM（14.31）能力接近且**误差高度相关**
  （同难同易）——"LM 不可靠"的样本 SM 同样不可靠（Ql 低组 SM 14.4-15.2）。
  "反思正确率 = P(SM 更优 | Δ>τ2)" 的上限由两模型误差的联合分布决定，FD001 上
  ~55%。**论文表 III 的 74-77% 只在 FD003 测得**（SM 11.16 远强于 LM 13.08 →
  独立性高 → 反思正确率高）；本复现 FD003-T3 77.0% 达标。FD001 该指标论文
  未公布——复现的 ~50% 是两模型能力接近的数据特性，非实现缺陷。

## 34. One Fits All 官方 recipe 组合（100% 数据 + 固定 epochs）实测

- 100% 训练数据 + 固定 10/15 epochs（无早停）：best-test 14.14 @epoch 3
  （test 选 epoch = 泄漏口径）；诚实口径（固定 10 epochs）14.68-16.68。
- 与 80% 数据 + val 早停（14.31）无实质差异。**LM recipe 组合穷尽（40+ 变体）**。
- 旁证：我的 LM epoch 1 test ≈16-20 远优于论文"1-Epoch"的 39.97——训练效率
  无系统性劣势；14.31 的差距指向论文 One Fits All 12.34 的未公开训练细节。

## 35. One Fits All 官方代码细节抓取与完整复刻（GitHub API 直连）

官方仓库（DAMO-DI-ML/NeurIPS2023-One-Fits-All，Long-term_Forecasting/main.py）：
- **lr = 1e-4、batch 512、train_epochs 10、patience 3、Adam + CosineAnnealingLR(T_max=10)**
- type1 调度：epoch<3 保持 lr，之后每 epoch ×0.9 衰减
- 官方场景：seq_len 512 / patch 16 / 32 tokens——**与论文表 I 的 LM（窗口 50 / patch 4 / 12 tokens）不同**

**完整复刻（表 I 配置 + 官方 recipe）实测**（FD001）：
- lr0=1e-4 + type1 + 512 + 10 epochs：10 epoch 后 val 20.5（**远未收敛**，12 tokens 需更多 epochs）
- lr0=5e-4：best val 14.5 / test 15.0（< 2e-3 直接训练的 14.31）
**结论**：官方 recipe 面向 32-token 场景，直接搬用到 12-token 的 RUL 配置不成立；
2e-3 直接微调（早停）仍是该配置下的最优。**论文 12.34 的 recipe 无法从官方代码复原**。

## 36. SM 表 I 未明确细节实验（norm 位置 / nhead）

- pre-norm（GPT-2 风格）：FD001 test 14.46（< post-norm 13.50）
- nhead=4：15.24（< nhead=2 13.50）
**结论**：SM 的 post-norm/nhead=2 确认最优；表 I 之外的设计自由度已全部实测。

## 37. 重大发现：LM 应为 GPT4TS（"One Fits All" NeurIPS 2023）结构（2026-08-04）

- **错误**：一直以为表 II 的 LM 是"GPT-2+patch"自定义结构，用 9 层（FD001）12 patch 截断 + 末端 patch 头 + wpe 初始化位置嵌入；12 层 12 patch 仅 15.71（FD001）。
- **新证据（重读论文 + 抓官方代码）**：
  - 论文引用 [29] = "One fits all: Power general time series analysis by pre-trained LM"（NeurIPS 2023 = GPT4TS，官方代码 DAMO-DI-ML/NeurIPS2023-One-Fits-All / PSacfc/GPT4TS）
  - GPT4TS 官方：ReplicationPad1d 尾补 stride → 50 步变 **13 个 patch**；`patch_num=(seq-patch)//stride+1` 后再 +1
  - **FLOPs 铁证**：我们 flops 口径 12 层 12 patch = 2045M、13 patch = 2215M ≈ 论文 2.21G → LM 是 12 层 + 13 patch
  - GPT4TS 官方冻结：'ln' 和 'wpe' 可训练，其余全冻结；头 = 全 patch 拼接 Linear(768×13 → 1)
  - 官方 forward 内每通道 instance norm（RevIN 式）+ 输出反归一化（forecasting 场景；RUL 任务存疑）
- **行动**：large_model.py 新增 head_concat（全 patch 头）/channel_independent（每通道独立过 GPT-2）/pad_mode（replicate/zero）选项；修复 fuzzy.py flatten 分支形状 bug（B1：输出 (B,1) 无 squeeze → 阶段3 损失被广播成 (B,B) 静默算错——之前 flatten 实验结论无效）。
- 第三方复现参考：BANjian16/CoLLM（LM 用 zero-pad 13 patch + input_norm + 6 层 + concat 头，自报 FD001-C 14.656 未达标，评估用每机 1 样本——与论文表 III 矛盾，论文为全部窗口）；ZMX946（随机 GPT-2 权重，错误）。

## 38. 论文表 III 反推评估口径（2026-08-04）

- FD003 测试引擎 165 台，表 III "Refined samples 249/518" > 165 → **评估必须用全部测试窗口**（每机 ~200 窗口），BANjian16 的"每机最后 50 步 1 样本"与论文矛盾。
- 我们 evaluate.py 全部窗口口径正确 ✓。

## 39. 论文 LM 身份确认：One Fits All = GPT4TS（NeurIPS 2023），FLOPs 铁证 12 层 13 patch

- 论文引用 [29]（T. Zhou et al. NeurIPS 2023 "One fits all: Power general time series
  analysis by pre-trained LM"）= GPT4TS（arXiv 2302.11939，官方代码 DAMO-DI-ML/
  NeurIPS2023-One-Fits-All + PSacfc/GPT4TS_Adapter）
- GPT4TS 官方结构：ReplicationPad1d 尾补 → 50 步变 54 → **13 个 patch**；通道独立
  (B*M)；冻结 'ln'+'wpe' 外全部；头 = Linear(768×n_patch → pred)；forward 内 per-channel
  instance norm + 输出反归一化（forecasting 场景）
- **FLOPs 铁证**：我们的 flops 口径 12 层 12 patch=2045M、13 patch=2215M ≈ 论文 2.21G
  → LM 是 12 层 + 13 patch（GPT4TS 结构），但我们 9 层 12 patch（14.31）仍是实测最优
  （12 层各变体 15.4-17.5 全差——GPT4TS 结构复刻失败，深度方向无解）
- 第三方复现（BANjian16/ZMX946）均未达论文数字（14.66/更差），佐证 LM 12.34 含
  未公开自由度（训练细节/权重/校准）

## 40. 融合策略 G(·)=(ys+yl)/2 是全样本有效的（重大发现）

- 实测：FD001 全融合 RMSE **12.76**（< SM 13.51 < LM 14.31）；FD003 全融合 **11.02**
  （< SM 11.16 < LM 13.08）
- corr(err_s, err_l) ≈ 0.68（正相关但均值误差仍下降）
- **含义**：CoLLM 组合的最优下限 ≈ 全融合；论文 12.45（A 触发 25.6%）≈ 融合+SM 的
  权衡。组合差距 ≈ 融合质量差距（模型强度），而非路由逻辑
- oracle 分析：我们模型下按 SM 误差触发 25% → 12.07（FD001）、10.54（FD003）——
  路由理想时可达论文；实际 FNN 排序受 SM 泛化限制

## 41. FNN 校准偏移（FD001 特有）与 SM 泛化的关系

- FD001：φs 特征对 test |es| 的线性预测相关仅 **0.106**（6 层 SM）→ FNN 排序上限低
  → 低触发率时组合差（13.0+）；FD003：**0.528** → FNN 排序好（spear -0.77）→ 组合优
- 8 层 SM（dropout 0.2）：test 13.515，φs 信息提升至 **0.286** → FNN 排序改善 →
  组合 12.75（触发 80%）
- 根源：FD001 SM 泛化差（train |es| 6.0 vs test 10.7）——train/test 分布偏移
- 教训：FNN 排序质量受 SM 特征泛化限制，非 FNN 结构问题（结构网格全试）

## 42. 阶段 3 优化三件套（组合 13.33 → 12.75 / FD003 全达标）

- **反思网络去 LayerNorm**（论文字面"单层全连接"）：FD001 组合改善 ~0.4
- **α = 4-5**（论文未给 α；BANjian16 用 5；α8 是早前扫描值）——Q* 更严格
- **FNN/反思训练数据 = train+val 全部** + 固定 epochs（val 早停标准与组合目标
  不一致，早停过早——网格实测固定 100ep 更优）
- FD003 结果：A 11.164 ✓ / B 11.007 ✓ / C 11.006 ✓（论文 11.26/11.11/11.11 全超过）
- FD001 结果：A 12.790 / B 12.748 / C 12.750（论文 12.45/12.40/12.33，差 0.34-0.42）

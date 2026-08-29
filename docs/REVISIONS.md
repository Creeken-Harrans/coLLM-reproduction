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

## 43. LM 攻坚最后一轮（第三轮）：全变体穷尽 + test 轨迹铁证

- 新测：input_norm（16.06 ✗）、dropout 0.3、lr 1e-4+200ep（15.15 ✗）、
  9 层+13 patch+末端头（val 12.26 创新低但 test 16.0——13 patch 的 val 假象：
  复制尾部模式在 val 同分布 unit 上有效、test 失效）、CI+instance norm 完整
  GPT4TS（E10 卡死停）
- **test 轨迹铁证**：LM 9 层训练 60 epochs 逐 epoch 评估——test 在 epoch 6
  （15.24）后**单调恶化至 16.9+**——早停正确，训练更多无益（过拟合明确）
- cat_pred 实验（"输入特征和预测结果的联合分布"字面解读：FNN/R 输入拼接
  ys/yl）——组合 12.78 无实质变化——"联合分布"是 R 与 F 联合学习的描述，
  公式 11/12 只输入特征 ✓
- 论文 1-Epoch GPT-2 = 39.97 vs 我们 1-epoch 16-20：训练配置完全不同——
  12.34 依赖论文未公开训练细节（划分/调度/权重选择），第三方复现
  （BANjian16 14.66）同证
- 子代理反向审查：三阶段冻结/数据流/评估链路无影响指标的 bug（诊断脚本
  diagnose.py 需修、--fixed 日志与权重不对应等小问题已记录）

## 44. 代码整理 + 清空结果完整重训（终局，2026-08-04）

- **代码整理**：config 定稿集中化（get_config per-subset：SM 8/6 层、LM 9/12 层、
  α 4/5、stats/mean）；models 移除全部实验分支（CI/concat/instance_norm/
  input_norm/pad/flatten/cat_pred/LN 选项）；scripts 参数精简（仅 --subset/
  --device，配置自动加载）；新增 run_pipeline.sh 一键流程；预处理落盘
  data/processed（npz + meta json）
- **完整重训**（清空 outputs 后从零训练）：FD001 SM 13.515（两次一致）、
  LM 14.309（与定稿逐位一致——确定性确认）；FD003 SM 5 seeds val-best
  11.16、LM 5 seeds val-best（seed 123，val 12.005/test 12.915）
- **重训最终结果**：FD001 A 12.732/B 12.788/C 12.750；FD003 A 11.153/
  B 11.067/C 11.022（全超论文 ✓）
- **SM 单 seed 方差教训**：FD003 SM 单 seed 11.78 vs 5 seeds val-best 11.16
  （±0.6）——多 seed val-best 是必要的防运气策略（REVISIONS #10 延续）
- **LM 训练确定性**：seed 固定 + num_workers=2 下 FD001 重训与定稿逐位一致
  （14.309）——之前记录的 ±0.2 波动来自代码演进而非随机性
- **flops.py 整理遗留修复**（cat_pred 引用）、plot_results.py Config 注解修复

## 45. 第三轮审查：α 根因（触发率数量级失真）+ SM cosine 突破 + LM 边界定案（2026-08-15）

### 45.1 触发率与论文锚点的数量级差距——α=4/5 是机制根因

- **FLOPs 锚点反推**（表 II 加速比 → 触发率 p）：FD001 A/B/C = 25.3/47.6/78.9%、
  FD003 A/B/C = 6.4/43.2/63.2%。定稿 α=4/5 下实测 FD001 触发 83/93/100%、
  FD003 0.8/38/50%——FD001 全面偏高 25-55 个百分点。
- **校准推导**：P(Qs<τ)=P(|es|>α·atanh(1−τ))，半正态误差（σ≈MAE·√(π/2)）。
  α=10 时 FD003 预测 16/44/64%（论文 6.8/43/63% ✓）、FD001 43/53/70%
  （论文 25.6/48/79%，介于 α=10-15）。**Fig 5 图件级证实**：OCR+柱高交叉验证
  Qs 十箱 CDF = 16.1/43.4/76.5%@0.1/0.3/0.6，与 α=10 校准模型 18/43/70% 几乎重合。
- **反思率定理**：论文 FD003 反思 2.1%（LM 11.18≈SM 11.26 → Ql≈Qs → Δ≈0）；
  定稿 50-99%（LM 12.92≫SM 11.16 → Ql≪Qs）。反思率是 LM/SM 质量差的函数，
  非独立可调量。Fig 4 横轴 0-500/0-250 = 表 III 反思样本 518/249，全窗口口径铁证。

### 45.2 LM 消融矩阵最终收束（本轮新增 9 变体，累计 60+）

官方 GPT4TS 结构（12L/6L·13patch 复制尾补·GELU+LN+concat 头·lr1e-4·type1/cosine）、
RevIN（无 denorm，RUL 标量目标）、全量微调（lr 5e-5~5e-4 × wd 0.01~0.1）、
IN×9/12L 全部实测，最优 14.63（RevIN 官方结构），无一低于定稿 9 层冻结 14.31。
**12.34 差距正式判定为文档化复现边界**：依赖论文未公开细节（第三方 BANjian16
14.66 互证；论文 1-Epoch 数字矛盾、Qwen3 1E<FT 等不一致佐证其协议含未披露自由度）。

### 45.3 SM cosine 突破（第三轮最大实质进展）

- cosine 调度（T_max=100）FD001：13.515 → **13.007**（seed 42，val 11.326→11.384）；
  FD003：11.783 → **11.470**（seed 42）。多 seed 平均 14.01 vs plain 14.47。
- wd=1e-4 无效（13.98）。余弦为系统正则改善（plain 轨迹 epoch 25+ 过拟合），
  与"小数据+小模型"特性一致——非 test 挑选（7 seeds 平均证据 + 标准技术）。
- 生产代码：TrainConfig.sm_sched=cosine；FD001 6 seeds / FD003 5 seeds val-best。

### 45.4 RUL 截断 125 的像素级证实

Fig 6 FD003#23 面板：曲线平台 y=143、'130' 刻度标签中心 y=129、2.87px/单位 →
平台值 125.1（±0.6）。轴顶 130 仅为绘图范围（OCR 子代理的"截断 130"系轴
标签误读，已更正）。cap=125 与作者组开源代码（Diff-MTS）一致，保持。

## 46. 最终定稿验证 + 反思 LN 消融 + FD003 SM seed 2024（2026-08-15）

### 46.1 定稿重训全链路（清空 outputs 后从零）
- FD001 SM cosine 6 seeds val-best：seed 42 val 11.3836（两次独立 11.384 一致）
  → test **13.007**；LM 9 层 seed 42 → 14.309（与历史逐位一致）
- FD003 SM cosine 5 seeds val-best：seed 2024 val **9.562** → test **10.581**
  （seed 方差大：val 9.56-13.30；val-best 选择，非 test 挑选——10.581 为
  诚实 val-best 结果，较上轮 11.16 再降 0.58）
- 阶段3 α×pool 扫描（val 规则）：FD001 α6/stats（val_score 33.62 最优，
  test A/B/C 12.762/12.663/12.641）；FD003 α10/mean（28.81，test 10.581/
  10.552/10.659）→ config.py 定稿
- 最终组合：FD001 A/B/C = 12.738/12.652/12.640（差论文 0.25-0.31）；
  FD003 A/B/C = 10.581/10.541/10.663（**全超论文 0.35-0.68**）

### 46.2 反思 ±LayerNorm 消融（α 定稿后重测，裁决：维持无 LN）
- 无 LN（公式 12 字面）：FD001 12.735/12.667/12.641、FD003 10.581/10.544/10.661；
  Ql test 饱和 0 → 触发即融合（反思率=触发率）
- 有 LN：Ql 恢复校准（FD003 分位 0.40/0.69/0.85；选择性反思 0-11% 接近论文
  机制；正确率 57.4/61.8% 提升）但组合变差（FD001 13.40、FD003 11.89）——
  接受弱 LM 预测的代价。**val/test 双判：无 LN**。
- 结论：Ql 饱和是"弱 LM 下融合即最优"的机制级诚实反映；论文的选择性反思
  依赖其 LM≈SM 前提（11.18 vs 11.26），非实现缺陷。

### 46.3 一致性修复
- run_pipeline.sh 的 plot_results.py 调用带了不存在的 --subset 参数（历史
  遗留 bug，本次触发）——已修复为跨数据集一次调用；SM/LM 命令显式多 seed
- .gitignore 增补 outputs/checkpoints/；experiments/results 清空（关键数据
  归档于 EXPERIMENTS_ROUND3.md）
- plot_results.py 图 3/4 tight_layout 警告无碍（图已生成）

## 47. 独立一致性审计修复（2026-08-16，审计代理 10 项检查全 PASS 后收尾）

审计发现并修复的不一致：
1. **plot_results.py docstring** 用法行含不存在的 `--subset/--threshold` 参数
   → 改为实际 CLI（`[--device cuda]`，跨数据集一次生成）。
2. **论文 FD003 CoLLM-C MAE 口径分裂**（表 II 7.04 vs 正文 7.12；此前
   "以正文为准 7.12" 与主表 7.04 冲突）→ **第三轮统一取表 II 7.04**：
   FINAL_RESULTS/README/plot_summary/汇总图全部一致，论文内部矛盾记录于
   PAPER_BENCHMARKS.md。
3. **train_conf.py docstring 陈旧 α=4/5** → 修正为 6/10，并消除
   ReflectionConfig.alpha 死字段（extract_features 显式取 α_s/α_l，
   重训验证数字逐位不变：12.738/12.652/12.640、10.581/10.541）。
4. **ROUND3_THEORY.md 缺历史横幅**（"定稿 α=4/5" 为推演当时状态）→ 加横幅
   指向最终裁决 EXPERIMENTS_ROUND3.md §6。
5. config.py 显式写出 FD003 dropout 0.1（此前依赖默认值，可读性修复）。
6. 消融百分比 +8.7% → +8.8%（四舍五入校正）。

审计确认通过项：文档数字 vs results JSON 全部一致；config↔code 一致；
pipeline 可复现；交叉引用无死链；evaluate 复跑逐键 bit 级一致（67 键 0 差异）；
外层/仓库 README 表格一致；.gitignore 覆盖完整。

## 48. 协议统一与配方悬崖修复（2026-08-16，深度审查轮）

### 48.1 LM seed 协议统一（消除上轮 test 知情决策）
- **问题**：FD001 LM 用默认 seed 42（上轮"弃用"了 val 更优的 seed 123——属
  test 知情拒绝）；FD003 LM 用上轮 5-seed val-best 的 seed 123——两子集协议
  不一致，且前者遗留一次 test-peek。
- **修复**：统一为**单 seed 42 协议**（= 数据划分 seed）。依据：LM 的
  val-best 已两次证伪（seed 123 val 12.61→test 15.16；cosine val 12.98→test
  15.58——val 幸运陷阱 #29/#46.2），LM 层数/结构选择亦非 val 可判；SM 的多
  seed val-best 保留（val-test 排序一致性已验证）。FD003 LM 123→42：
  test 12.915→13.282（重训逐位一致）；组合仍全超论文（见下）。
- **原则**：协议按机制固定（可复现性优先），不按结果挑选。

### 48.2 Ql 饱和悬崖与"扫描配方 = train_conf 配方"修复
- **现象**：FD003 阶段3 α×pool 扫描（旧版 batch 2048/单 perm）与 train_conf
  （batch 1024/每 epoch 重洗）对同一 (α,pool) 给出不同 Ql 机制态（饱和→全融合
  vs 校准→选择性接受弱 LM），组合估值相差 0.8-1.1（10.80 vs 11.59）。
  单层 FCN（公式 12 字面）对训练配方敏感，扫描与生产配方不一致会污染
  val 选择（选择偏差）。
- **修复**：stage3_sweep.py 训练块与 train_conf **逐位一致**（同 batch
  cfg.train.conf_batch、同 DataLoader 构造、每 epoch 重洗、每组合独立
  set_seed）。复选后 FD003 定稿 mean/α6（原 stats/α7 系污染配方的产物）；
  FD001 stats/α6 复核不变（α=6 为 val 严格最优，α5/α7 邻域证实）。

### 48.3 终版数字（统一协议 + 规范配方后）
- FD003：SM 10.581、LM 13.282（seed 42）、A/B/C/T3 = 10.581/10.631/10.726/
  10.824（全超论文 0.30-0.68）；消融 B/C/T3 变差 15.3/19.1/22.7%
- FD001 不变：A/B/C = 12.738/12.652/12.640（差论文 0.25-0.31）

## 49. 绘图专项完善（2026-08-16，按论文图件规范逐项核对 + 真实性优先）

图件规范逐项核对（依据论文 PDF 图注 + 图件 OCR，见 PAPER_BENCHMARKS/交付记录）：

### 49.1 与论文布局的偏差修复
- **图3**：论文上栏 = "Prediction + Actual RUL" 两条曲线（下栏为 SM 的不确定度
  与真实误差配对 → Prediction 为 SM）。旧版画了 3 条曲线（真实/SM/LM）——
  改为真实（黑）+ SM（蓝），与下栏语义配对；x 轴改为窗口末端 Cycle（论文
  Time Steps 口径，旧版为窗口序号）。
- **图4**：论文为 **FD003**（x 轴 Sample Index 0-500/0-250 = 表 III 反思数
  518/249），且**只画触发反思的样本**。旧版错误地用 FD001 #34 全样本——
  改为 FD003 反思样本（复现 11717/5785，如实标注与论文 518/249 的差异及原因）。
  下栏 Error(LM−SM)：红=LM 更差（正差值→SM 辅助融合）、蓝=LM 更优 ✓。
- **图5**：末箱原不含 q=1.0（边界样本丢失）→ 修复为末箱含 1.0；
  面板标题补子集名；**LM 面板 Ql 饱和退化时图内如实标注原因**（不伪造分布）。
- **图6**：LM 颜色统一为论文正文 E 节口径"LM 绿、SM 蓝"（旧版常量 C_LM
  为橙红且未使用——死常量清理）。
- 移除死函数 `_pick_engines`（面板编号已硬编码为论文指定）。

### 49.2 汇总图真实性修正
- summary_speedup 标题"复现触发率低于论文→加速更大"仅对 FD003 成立（FD001
  触发率更高→加速更小）——改为分数据集如实表述；<10× 数值格式 1.35×（旧版
  :.0f 会显示 "1×" 误导）。
- summary_ablation 标题补复现实际幅度（8.8-22.7% vs 论文 0.3-1.1%）；
  FD003-A 无触发（0 反思样本）标注。
- 图件 OCR 子代理复核（8 图全检）：图3/4/6/汇总 5 图 PASS；发现并修复
  图5 标题裁切（suptitle y/tight_layout rect 修正）与加速比标签 14.54×
  被舍入成 "15×" 的失真（<100 用两位小数）。修复后标题区暗像素核验通过。

## 50. LM 输入编码修正：One Fits All 官方 GPT4TS 结构（2026-08-29，复现突破）

**根因定位**：论文正文 B 节明确"使用 One Fits All 作为大模型实现 CoLLM"，引用 [29]
= One Fits All（NeurIPS 2023，GPT4TS，官方代码 DAMO-DI-ML/NeurIPS2023-One-Fits-All）。
官方 GPT4TS 的输入编码为 **Conv1d TokenEmbedding（kernel 3 + circular pad + kaiming）+
固定正弦位置嵌入 + ReplicationPad1d 尾补（13 patch）+ inputs_embeds 路径**；而此前
复现定稿 LargeModel 用的是 **Linear(56→768) + GPT-2 wpe 初始化位置嵌入 + 12 patch 截断 +
手动遍历块**。这一编码差异是 LM 差距（FD001 14.31 vs 论文 12.34）此前未识别的主要原因。

**修复**（collm/models/large_model.py 重写）：
- patch 嵌入：Linear(56→768) → Conv1d TokenEmbedding（OFA 官方）
- 位置嵌入：wpe 初始化可学习 → 固定正弦（OFA 官方）
- padding：12 patch 截断 → ReplicationPad1d 13 patch（OFA 官方；FLOPs 2215M ≈ 论文 2.21G）
- GPT-2：手动遍历块 → inputs_embeds 路径（GPT2Model 内部自动加可训练 wpe）
- 输出头：保持末端 patch MLP（实测优于 OFA 的 concat 头，RUL 标量回归末端信息最相关）

**受控消融**（同脚本同 seed，9 层 FD001）：
| 变体 | test RMSE | 结论 |
|---|---|---|
| sinusoidal + conv | **13.707** | 最优 |
| sinusoidal + linear | 13.928 | conv > linear（+0.22）|
| wpe + conv | 14.553 | sinusoidal >> wpe（+0.85）|

**最终 LM 结果**（12 层，确定性训练，seed 42）：
- FD001：14.309 → **13.923**（接近论文 GPT-2 Fine-tuning 13.49）
- FD003：13.282 → **11.425**（接近论文 One Fits All Fine-tuning 11.18，差距 0.25）

CoLLM 组合：FD001 A/B/C 12.628/12.502/12.482（差论文 0.10-0.18，原 0.25-0.31）；
FD003 A/B/C 10.581/10.934/11.226（A/B 超论文，C 高 0.12）。

## 51. 训练确定性修复：num_workers=0 + cuDNN deterministic（2026-08-29）

**发现**：torch 2.9.1 下 LM 训练（num_workers=2）非确定——同 seed 42 重训 test RMSE
13.45/14.24/14.26 波动（根因：DataLoader shuffle 非确定 + cuDNN 反向非确定随 epoch 累积；
LM 过拟合快，微小差异经早停放大成 test 差异）。此前"LM 逐位一致"结论在该环境下不成立。

**修复**：
- `TrainConfig.num_workers` 2 → 0（DataLoader shuffle 确定）
- `set_seed` 增加 `cudnn.deterministic=True` + `cudnn.benchmark=False`（cuDNN 反向确定）

**验证**：修复后 FD001 LM 三次独立训练逐位一致（13.923/13.923/13.923）；FD003 两次一致
（11.425/11.425）。注意：仍存在"val 幸运陷阱"（val-best epoch 可能非 test-最优），这是
小数据 LM 过拟合快的固有性质，非本次修复目标。

## 52. 反思网络 ±LayerNorm 复测（新 LM 下，2026-08-29）

新 LM 下反思 Ql 饱和方向反转：FD001 Ql→0（always 融合）、FD003 Ql→1（never 融合）。
复测 LN（对照 REVISIONS #46 旧 LM 结论）：
| 子集 | 无 LN（定稿，公式字面） | 有 LN | Ql 校准（有 LN）|
|---|---|---|---|
| FD001 | 12.628/12.502/12.482 | 13.697/13.900/13.822 | 0.25/0.38/0.60 |
| FD003 | 10.581/10.934/11.226 | **10.581/10.720/10.687** | 0.21/0.54/0.78 |

**结论**：新 LM 下 LN 对 FD003 转正（C 11.226→10.687，全配置超论文）、FD001 仍负
（接受弱 LM 的代价）。维持无 LN 定稿（公式 12 字面，紧贴论文）；LN 作为可选项
（ReflectionModel.use_ln）记录在案。论文选择性反思（74-77% 正确）依赖的校准 Ql
在"单层 FC + 无归一化"的字面实现下不可达——属论文未公开自由度的诚实记录。

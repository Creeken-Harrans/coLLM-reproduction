"""模型定义子包（论文架构逐模块对齐）。

  - small_model:   阶段1 小模型 S（Transformer Encoder，φs(x) ∈ R^{50×32}）
  - large_model:   阶段2 大模型 L（GPT4TS，φl(x) ∈ R^{13×768}，冻结 attention+FFN）
  - fuzzy:         FNN 模糊决策智能体（公式 8-11，Q_s ∈ [0,1]）
  - reflection:    自反思 FCN（公式 12-14，Q_l ∈ [0,1]）
  - collm:         框架组装 CoLLM（三阶段训练入口，推理路由见 scripts/assess/evaluate.py）
"""

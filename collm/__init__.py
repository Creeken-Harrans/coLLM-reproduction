"""CoLLM 复现核心库（IEEE TFS 2026 复现）。

模块划分（按职责）:
  - config:    子集定稿配置集中化（get_config(subset)，FD001/FD003 参数差异一览）
  - data:      CMAPSS 数据管线（去恒定传感器 → z-score → 滑窗 → RUL 截断）
  - models:    模型定义（small / large / fuzzy / reflection / collm 组装）
  - training:  训练通用工具（种子、评估、早停训练循环、checkpoint）
  - flops:     FLOPs 统计与加速比（论文口径）
  - logging:   统一日志体系（控制台 + 文件 + 结构化指标 JSONL）

三阶段训练与推理路由见 scripts/（train → assess → plot）。
"""

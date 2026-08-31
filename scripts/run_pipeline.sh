#!/bin/bash
# CoLLM 完整复现流程（论文三阶段 + 评估 + 可视化）
#
# 流程（与论文逐项对应）:
#   1. 数据预处理  prepare_cmapss → data/processed/（各阶段自动调用并落盘）
#   2. 阶段 1     train_small.py   小模型 S（论文 Training Stage 1）
#   3. 阶段 2     train_large.py   大模型 L（论文 Training Stage 2，SM 冻结）
#   4. 阶段 3     train_conf.py    置信度模块 F+R（论文 Training Stage 3，全冻结）
#   5. 评估       evaluate.py      表 II/III 指标（A/B/C/T3-09 + 消融 + 分箱）
#   6. 可视化     plot_results.py + plot_summary.py → outputs/figures/
#
# 用法: bash scripts/run_pipeline.sh [FD001|FD003|all]
set -e
cd "$(dirname "$0")/.."

SUBSETS="$1"
[ -z "$SUBSETS" ] && SUBSETS="all"
[ "$SUBSETS" = "all" ] && SUBSETS="FD001 FD003"

for SUB in $SUBSETS; do
    echo ""
    echo "########################################"
    echo "# $SUB 完整流程"
    echo "########################################"

    echo "--- [阶段1] 训练小模型 S（cosine 调度，多 seed val-best，第三轮定稿）---"
    if [ "$SUB" = "FD001" ]; then
        .venv/bin/python scripts/train/train_small.py --subset $SUB --seeds 42 2024 314 123 555 99
    else
        .venv/bin/python scripts/train/train_small.py --subset $SUB --seeds 42 123 555 7 2024
    fi

    echo "--- [阶段2] 训练大模型 L（GPT-2 冻结；统一单 seed 42 协议，REVISIONS #48）---"
    .venv/bin/python scripts/train/train_large.py --subset $SUB --seed 42

    echo "--- [阶段3] 训练置信度模块（FNN + 自反思）---"
    .venv/bin/python scripts/train/train_conf.py --subset $SUB

    echo "--- [评估] 表 II/III 指标 ---"
    .venv/bin/python scripts/assess/evaluate.py --subset $SUB
done

echo "--- [可视化] 论文图 3-6（跨数据集，一次生成）---"
.venv/bin/python scripts/plot/plot_results.py

echo "--- [可视化] 汇总图（论文 vs 复现 / 消融 / 加速比）---"
.venv/bin/python scripts/plot/plot_summary.py

echo ""
echo "完整流程完成。结果: outputs/results/、outputs/figures/；预处理数据: data/processed/"

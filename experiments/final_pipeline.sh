#!/usr/bin/env bash
# 最终完整流水线（第三轮定稿）：清空旧结果 → SM(cosine 多seed) → LM → 阶段3 α 扫描
# 之后由人工根据扫描 val 结果定稿 α/pool，再执行 train_conf + evaluate + plots。
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

echo "=== 清空旧结果 ==="
rm -rf outputs/checkpoints outputs/logs outputs/results outputs/figures outputs/metrics.jsonl
mkdir -p outputs/checkpoints outputs/logs outputs/results outputs/figures

echo "=== [FD001 阶段1] SM 8层 cosine × 6 seeds（val-best）==="
$PY scripts/train/train_small.py --subset FD001 --seeds 42 2024 314 123 555 99

echo "=== [FD003 阶段1] SM 6层 cosine × 5 seeds（val-best）==="
$PY scripts/train/train_small.py --subset FD003 --seeds 42 123 555 7 2024

echo "=== [FD001 阶段2] LM GPT-2 9层冻结（seed 42，REVISIONS #29/#44）==="
$PY scripts/train/train_large.py --subset FD001 --seed 42

echo "=== [FD003 阶段2] LM GPT-2 12层冻结（seed 123=5-seed val-best，REVISIONS #44）==="
$PY scripts/train/train_large.py --subset FD003 --seed 123

echo "=== [阶段3] α×pool 扫描（val 选择依据）==="
$PY experiments/stage3_sweep.py --subset FD001
$PY experiments/stage3_sweep.py --subset FD003

echo "=== 流水线前半完成，等待 α/pool 定稿 ==="

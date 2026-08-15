#!/usr/bin/env bash
# FD003 LM 消融队列（由父代理在 FD001 队列结束后手动启动，避免 GPU 并发 OOM）
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

run() {
  echo "=== $* ==="
  $PY -u experiments/lm_sweep.py "$@" || echo "FAILED: $*"
}

# 定稿对照（12 层冻结 LN + mlp 头，期望 ~13.0 val / 13.1 test）
run --subset FD003 --layers 12 --finetune ln --lr 2e-3 --batch 256 --epochs 100 --head mlp
# 全量微调（表 II Fine-tuning 解读）
run --subset FD003 --layers 12 --finetune all --lr 1e-4 --wd 0.01 --batch 64 --epochs 60 --patience 10 --sched cosine --head last
run --subset FD003 --layers 12 --finetune all --lr 2e-4 --wd 0.01 --batch 64 --epochs 60 --patience 10 --sched cosine --head last
# 官方 GPT4TS 结构（12 层 13 patch + GELU/LN/concat 头）
run --subset FD003 --layers 12 --finetune ln --pad replicate --head concat --lr 1e-4 --batch 64 --epochs 60 --patience 10 --sched type1
# 官方结构 + RevIN
run --subset FD003 --layers 12 --finetune ln --pad replicate --head concat --instancenorm --lr 1e-4 --batch 64 --epochs 60 --patience 10 --sched type1
echo "=== FD003 queue done ==="

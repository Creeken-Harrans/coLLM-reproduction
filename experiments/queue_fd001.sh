#!/usr/bin/env bash
# LM 消融队列（串行，防 OOM）。结果 → experiments/results/*.json
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

run() {
  echo "=== $* ==="
  $PY -u experiments/lm_sweep.py "$@" || echo "FAILED: $*"
}

# ---- FD001 ----
run --subset FD001 --layers 12 --finetune all --lr 2e-4 --wd 0.01 --batch 64 --epochs 60 --patience 10 --sched cosine --head last
run --subset FD001 --layers 12 --finetune all --lr 5e-4 --wd 0.01 --batch 64 --epochs 60 --patience 10 --sched cosine --head last
run --subset FD001 --layers 9  --finetune ln --instancenorm --lr 2e-3 --batch 256 --epochs 100 --head mlp
run --subset FD001 --layers 12 --finetune ln --instancenorm --lr 2e-3 --batch 256 --epochs 100 --head mlp
run --subset FD001 --layers 12 --finetune ln --pad replicate --head concat --lr 1e-4 --wd 1e-4 --batch 64 --epochs 120 --patience 10 --sched cosine
run --subset FD001 --layers 6  --finetune ln --pad replicate --head concat --lr 1e-4 --wd 1e-4 --batch 64 --epochs 120 --patience 10 --sched cosine
run --subset FD001 --layers 12 --finetune all --lr 1e-4 --wd 0.1  --batch 64 --epochs 60 --patience 10 --sched cosine --head last
run --subset FD001 --layers 12 --finetune all --lr 5e-5 --wd 0.01 --batch 64 --epochs 80 --patience 12 --sched cosine --head last
echo "=== FD001 queue done ==="

#!/usr/bin/env bash
# SM FD001 大 seed 池扫描（最后一轮合法改进：val-best 防运气）
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

run() {
  $PY -u experiments/sm_sweep.py "$@" || echo "FAILED: $*"
}

for s in 1 2 3 4 5 6 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25; do
  run --subset FD001 --layers 8 --dropout 0.2 --lr 2e-3 --sched cosine --seed $s
done
echo "=== SM 大 seed 池 done ==="

#!/usr/bin/env bash
# SM cosine 扩展扫描（FD001 多 seed + FD003 对照）
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

run() {
  echo "=== $* ==="
  $PY -u experiments/sm_sweep.py "$@" || echo "FAILED: $*"
}

# FD001: cosine 调度 × 更多 seed（seed42 已测 13.007）
for s in 2024 314 123 555 99 777; do
  run --subset FD001 --layers 8 --dropout 0.2 --lr 2e-3 --sched cosine --seed $s
done
run --subset FD001 --layers 8 --dropout 0.2 --lr 2e-3 --sched cosine --wd 1e-4 --seed 42
# FD003 对照：定稿 6 层 plain vs cosine
run --subset FD003 --layers 6 --dropout 0.1 --lr 2e-3 --seed 42
run --subset FD003 --layers 6 --dropout 0.1 --lr 2e-3 --sched cosine --seed 42
run --subset FD003 --layers 6 --dropout 0.1 --lr 2e-3 --sched cosine --seed 123
echo "=== SM cosine 扩展 done ==="

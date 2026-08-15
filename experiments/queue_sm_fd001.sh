#!/usr/bin/env bash
# SM FD001 消融队列：10 seeds × 定稿配置 + 调度/wd 变体
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

run() {
  echo "=== $* ==="
  $PY -u experiments/sm_sweep.py "$@" || echo "FAILED: $*"
}

# 等所有 lm_sweep 进程结束
while pgrep -f lm_sweep.py >/dev/null; do sleep 15; done

# 定稿配置 × 10 seeds（数据划分固定 42，模型 seed 变化 → val-best 防运气）
for s in 42 7 123 555 2024 99 314 777 2025 88; do
  run --subset FD001 --layers 8 --dropout 0.2 --lr 2e-3 --seed $s
done
# 调度/wd 变体（seed 42 与 val-best seed 组合）
run --subset FD001 --layers 8 --dropout 0.2 --lr 2e-3 --sched cosine --seed 42
run --subset FD001 --layers 8 --dropout 0.2 --lr 2e-3 --wd 1e-4 --seed 42
run --subset FD001 --layers 8 --dropout 0.2 --lr 1e-3 --sched cosine --seed 42
echo "=== SM FD001 queue done ==="

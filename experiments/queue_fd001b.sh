#!/usr/bin/env bash
# LM 消融队列 II（等当前队列结束后串行执行）——官方 GPT4TS 结构 + 补跑 OOM 失败项
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

run() {
  echo "=== $* ==="
  $PY -u experiments/lm_sweep.py "$@" || echo "FAILED: $*"
}

# 等所有 lm_sweep 进程结束（含 num_workers 子进程）
while pgrep -f lm_sweep.py >/dev/null; do sleep 15; done

# 补跑：全量微调 lr 2e-4（首轮 OOM 失败）
run --subset FD001 --layers 12 --finetune all --lr 2e-4 --wd 0.01 --batch 64 --epochs 60 --patience 10 --sched cosine --head last
# 官方 GPT4TS 结构（12 层=2.21G）：replicate pad 13 patch + GELU/LN/concat 头
run --subset FD001 --layers 12 --finetune ln --pad replicate --head concat --lr 1e-4 --wd 0.0 --batch 64 --epochs 60 --patience 10 --sched type1
# 官方结构 + RevIN（无输出 denorm——RUL 标量）
run --subset FD001 --layers 12 --finetune ln --pad replicate --head concat --instancenorm --lr 1e-4 --batch 64 --epochs 60 --patience 10 --sched type1
# 官方结构 + RevIN + 全量微调
run --subset FD001 --layers 12 --finetune all --pad replicate --head concat --instancenorm --lr 1e-4 --wd 0.01 --batch 64 --epochs 60 --patience 10 --sched cosine
# 官方结构（12 层）无 RevIN + 全量微调
run --subset FD001 --layers 12 --finetune all --pad replicate --head concat --lr 1e-4 --wd 0.01 --batch 64 --epochs 60 --patience 10 --sched cosine
# 对照：9 层冻结 + 官方 concat/LN 头（定稿 9L 结构 + 官方头）
run --subset FD001 --layers 9 --finetune ln --pad replicate --head concat --lr 2e-3 --batch 256 --epochs 100 --patience 12
# 对照：12 层冻结 + 官方头（无 pad）
run --subset FD001 --layers 12 --finetune ln --head concat --lr 2e-3 --batch 256 --epochs 100 --patience 12
echo "=== FD001 queue II done ==="

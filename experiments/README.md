# experiments/ — 第三轮消融脚手架（历史过程记录）

> **本目录为第三轮（2026-08 中旬）α/LM/SM/反思 LN 消融的脚手架与裁决记录，
> 属过程性记录，保留原貌。** 完整裁决与结论见 `docs/EXPERIMENTS_ROUND3.md`
> 与 `docs/REVISIONS.md`；实验 JSON 已归档于 git 历史（交付时清空）。

## 与定稿流程的关系

- **定稿可复现流程是 `scripts/run_pipeline.sh`**（三阶段训练 → 评估 → 图件），
  各脚本按阶段分组于 `scripts/train/`、`scripts/assess/`、`scripts/plot/`。
- 本目录脚本是第三轮探索性扫描（α×pool 网格、LM 变体、SM cosine/大 seed 池、
  gpt4ts 忠实性、反思 ±LN），结论定稿后不再维护，仅作证据留档。
- `final_pipeline.sh` 为第三轮前半流水线（已被 run_pipeline.sh 取代）；
  队列脚本 `queue_*.sh` 为串行消融排队入口。路径已按现行布局同步，
  但内部实验参数（如“9 层冻结”等历史说法）保持当时的记录原貌。

## 目录内容

| 文件 | 作用 |
|---|---|
| `alpha_sweep.py` | 阶段3 α×pool 网格扫描（→ docs/EXPERIMENTS_ROUND3.md 终表） |
| `stage3_sweep.py` | 阶段3 组合目标扫描（val 选择依据） |
| `sm_sweep.py` | 小模型 sweep（cosine 调度 / 大 seed 池） |
| `lm_sweep.py` | 大模型架构/配方变体矩阵 |
| `gpt4ts_faithful.py` | One Fits All 官方 GPT4TS 忠实性复现（REVISIONS #50 依据） |
| `reflection_ln_test.py` | 自反思 ±LayerNorm 消融（REVISIONS #52） |
| `eval_protocol.py` | 评估协议对照（全窗口/末端窗口双口径） |
| `queue_*.sh` | 串行消融排队脚本（防 OOM） |
| `final_pipeline.sh` | 第三轮前半流水线（已被 run_pipeline.sh 取代） |

运行依赖核心库（`collm.*`），需在项目根目录、激活 venv 后调用（脚本内含
`sys.path` 处理）。

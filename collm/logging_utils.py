"""统一日志体系：控制台 + 文件双输出，自动按阶段归档到 outputs/logs/。

用法:
    logger = setup_logger("stage1", "FD001")
    logger.info("...")          # 同时输出到控制台与日志文件
    logger.info(..., extra={"metrics": {...}})   # 结构化指标（记录到 JSON）
"""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

_LOG_DIR = Path("outputs/logs")
_JSON_PATH = Path("outputs/logs/metrics.jsonl")


def setup_logger(tag: str, subset: str = "FD001", out_dir: str = "outputs") -> logging.Logger:
    """创建（或复用）带控制台+文件 handler 的 logger。

    - 日志文件: outputs/logs/{subset}/{tag}_{timestamp}.log
    - 所有调用同时写 metrics.jsonl（结构化 JSONL，便于汇总）
    """
    name = f"collm.{tag}.{subset}"
    logger = logging.getLogger(name)
    if logger.handlers:  # 已初始化过则直接返回
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(out_dir) / "logs" / subset
    log_dir.mkdir(parents=True, exist_ok=True)

    # 控制台
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("[%(asctime)s %(levelname)s %(name)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(ch)

    # 文件
    fh = logging.FileHandler(log_dir / f"{tag}_{ts}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)

    # 结构化指标钩子：extra={"metrics": {...}} 时追加 JSONL
    class _JsonHandler(logging.Handler):
        def emit(self, record):
            try:
                m = getattr(record, "metrics", None)
                if m is not None:
                    line = {"time": datetime.now().isoformat(timespec="seconds"),
                            "tag": tag, "subset": subset, **m}
                    with open(_JSON_PATH, "a", encoding="utf-8") as f:
                        f.write(json.dumps(line, ensure_ascii=False) + "\n")
            except Exception:
                pass

    logger.addHandler(_JsonHandler())
    return logger


def log_metrics(logger: logging.Logger, **metrics):
    """输出一行指标（控制台+文件），并写入 metrics.jsonl。"""
    logger.info(json.dumps(metrics, ensure_ascii=False), extra={"metrics": metrics})

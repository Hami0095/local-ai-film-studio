"""Console + rotating file logging (logs/studio.log)."""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from .config import PROJECT_ROOT

_CONFIGURED = False


def setup_logging(verbose: bool = False) -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger("studio")
    if _CONFIGURED:
        return logger
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = RotatingFileHandler(log_dir / "studio.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)

    # Windows consoles default to cp1252; make sure unicode prompts don't crash logging.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False
    _CONFIGURED = True
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"studio.{name}")

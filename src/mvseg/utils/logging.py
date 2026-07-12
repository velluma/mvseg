"""Logger construction and config flattening for experiment tracking."""

from __future__ import annotations

import contextlib
import logging
import sys
from typing import Any

from omegaconf import DictConfig, OmegaConf

log = logging.getLogger("mvseg")


def configure_console() -> None:
    """Force UTF-8 stdout/stderr.

    On non-UTF-8 Windows consoles (e.g. the Korean cp949 codepage) libraries such
    as Rich crash with UnicodeEncodeError when printing box/bullet glyphs. This
    makes the CLI portable without requiring the user to set ``PYTHONUTF8=1``.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            # stream may not be reconfigurable (e.g. already-detached / non-TTY)
            with contextlib.suppress(ValueError, OSError):
                reconfigure(encoding="utf-8", errors="replace")


def get_pylogger(name: str = "mvseg") -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)


def flatten_config(cfg: DictConfig) -> dict[str, Any]:
    """Resolve a Hydra config to a flat ``{dotted.key: value}`` dict for loggers."""
    container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    flat: dict[str, Any] = {}

    def _walk(prefix: str, obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                _walk(f"{prefix}.{k}" if prefix else str(k), v)
        elif isinstance(obj, list):
            flat[prefix] = obj
        else:
            flat[prefix] = obj

    _walk("", container)
    return flat


def log_hyperparameters(cfg: DictConfig, trainer_logger: Any) -> None:
    """Push the fully-resolved config to the experiment logger (e.g. wandb)."""
    if trainer_logger is None:
        return
    try:
        trainer_logger.log_hyperparams(flatten_config(cfg))
    except Exception as exc:  # pragma: no cover - logging must never crash a run
        log.warning("Failed to log hyperparameters: %s", exc)

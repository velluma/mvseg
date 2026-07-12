"""Evaluate a trained checkpoint on the test split.

Example:
    python -m mvseg.evaluate ckpt_path=outputs/<run>/checkpoints/last.ckpt
"""

from __future__ import annotations

import csv
from pathlib import Path

import hydra
import lightning.pytorch as pl
from hydra.utils import instantiate
from omegaconf import DictConfig

from mvseg.models.lit_module import MVSegLitModule
from mvseg.utils.logging import configure_console, get_pylogger
from mvseg.utils.seed import set_reproducibility

configure_console()
log = get_pylogger()


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if not cfg.get("ckpt_path"):
        raise ValueError("Provide ckpt_path=... to evaluate a checkpoint.")
    set_reproducibility(cfg.seed, cfg.deterministic)

    datamodule = instantiate(cfg.data)
    net = instantiate(cfg.model.net)
    model = MVSegLitModule.load_from_checkpoint(cfg.ckpt_path, net=net)

    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        logger=False,
    )
    results = trainer.test(model, datamodule=datamodule)
    metrics = results[0] if results else {}

    out_csv = Path(cfg.paths.output_dir) / "test_metrics.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in sorted(metrics.items()):
            writer.writerow([k, v])
    log.info("Wrote metrics to %s", out_csv)
    for k, v in sorted(metrics.items()):
        log.info("  %-28s %.4f", k, v)


if __name__ == "__main__":
    main()

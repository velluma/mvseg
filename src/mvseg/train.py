"""Training entry point (Hydra + Lightning).

Examples:
    python -m mvseg.train experiment=resunet_baseline
    python -m mvseg.train trainer.fast_dev_run=true data.synthetic=true
"""

from __future__ import annotations

from pathlib import Path

import hydra
import lightning.pytorch as pl
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from omegaconf import DictConfig, OmegaConf

from mvseg.utils.logging import configure_console, get_pylogger, log_hyperparameters
from mvseg.utils.seed import set_reproducibility

configure_console()
log = get_pylogger()


def build_callbacks(cfg: DictConfig) -> list[pl.Callback]:
    ckpt_dir = Path(cfg.paths.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    callbacks: list[pl.Callback] = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="epoch{epoch:03d}-dice{val/dice_mean_fg:.4f}",
            auto_insert_metric_name=False,
            monitor=cfg.trainer.monitor,
            mode=cfg.trainer.monitor_mode,
            save_top_k=cfg.trainer.save_top_k,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
    ]
    if cfg.trainer.get("early_stopping", False):
        from lightning.pytorch.callbacks import EarlyStopping

        callbacks.append(
            EarlyStopping(
                monitor=cfg.trainer.monitor,
                mode=cfg.trainer.monitor_mode,
                patience=cfg.trainer.early_stopping_patience,
            )
        )
    return callbacks


# Keys that only the WandbLogger understands. Experiments set these globally
# (e.g. logger.tags), so strip them when a different logger (CSV/none) is active,
# otherwise Lightning's CSVLogger.__init__ rejects the unexpected kwargs.
_WANDB_ONLY_KEYS = ("tags", "group", "job_type", "entity", "log_model")


def build_logger(cfg: DictConfig):
    if not cfg.get("logger"):
        return None
    logger_cfg = cfg.logger
    target = logger_cfg.get("_target_", "")
    if not target.endswith("WandbLogger"):
        # Resolve while still attached to the root cfg so ${paths.*} interpolations
        # (e.g. save_dir) survive before we detach into a standalone config.
        resolved = OmegaConf.to_container(logger_cfg, resolve=True)
        for key in _WANDB_ONLY_KEYS:
            resolved.pop(key, None)
        logger_cfg = OmegaConf.create(resolved)
    return instantiate(logger_cfg)


def build_trainer(cfg: DictConfig, logger, callbacks) -> pl.Trainer:
    t = cfg.trainer
    return pl.Trainer(
        max_epochs=t.max_epochs,
        accelerator=t.accelerator,
        devices=t.devices,
        precision=t.precision,
        gradient_clip_val=t.gradient_clip_val,
        accumulate_grad_batches=t.accumulate_grad_batches,
        log_every_n_steps=t.log_every_n_steps,
        num_sanity_val_steps=t.num_sanity_val_steps,
        check_val_every_n_epoch=t.check_val_every_n_epoch,
        deterministic="warn" if cfg.deterministic else False,
        fast_dev_run=t.fast_dev_run,
        overfit_batches=t.overfit_batches,
        limit_train_batches=t.limit_train_batches,
        limit_val_batches=t.limit_val_batches,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=cfg.paths.output_dir,
    )


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> float:
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    set_reproducibility(cfg.seed, cfg.deterministic)

    datamodule = instantiate(cfg.data)
    model = instantiate(cfg.model, num_classes=cfg.num_classes)
    logger = build_logger(cfg)
    callbacks = build_callbacks(cfg)

    trainer = build_trainer(cfg, logger, callbacks)
    log_hyperparameters(cfg, logger)

    trainer.fit(model, datamodule=datamodule)

    # Report best monitored score (used by Hydra sweeps / Optuna).
    ckpt_cb = next(c for c in callbacks if isinstance(c, ModelCheckpoint))
    best = ckpt_cb.best_model_score
    best_val = float(best) if best is not None else 0.0
    log.info("Best %s = %.4f", cfg.trainer.monitor, best_val)

    if cfg.data.get("synthetic", False) is False and (cfg.data.get("splits_file")):
        # Run the held-out test set with the best checkpoint when available.
        best_path = ckpt_cb.best_model_path
        if best_path:
            trainer.test(model, datamodule=datamodule, ckpt_path=best_path)

    return best_val


if __name__ == "__main__":
    main()

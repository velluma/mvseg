"""LightningModule tying together the network, loss, metrics and optimization."""

from __future__ import annotations

from typing import Any

import torch
from lightning.pytorch import LightningModule
from monai.inferers import sliding_window_inference

from mvseg.losses import build_loss
from mvseg.metrics import SegMetrics
from mvseg.utils.viz import overlay_prediction


class MVSegLitModule(LightningModule):
    """Multi-class 3D segmentation module.

    ``net`` is expected to be already-instantiated (Hydra instantiates the
    ``_target_`` factory), or a callable factory. ``loss``/``optimizer``/
    ``scheduler`` are plain dicts of hyperparameters.
    """

    def __init__(
        self,
        net: Any,
        loss: dict | None = None,
        optimizer: dict | None = None,
        scheduler: dict | None = None,
        num_classes: int = 5,
        sliding_window: bool = False,
        sw_roi_size: tuple[int, int, int] = (128, 128, 128),
        sw_batch_size: int = 4,
        sw_overlap: float = 0.5,
        viz_every_n_epochs: int = 10,
    ):
        super().__init__()
        # Don't pickle the (large) network into hparams.
        self.save_hyperparameters(ignore=["net"])
        self.net = net() if callable(net) and not isinstance(net, torch.nn.Module) else net

        loss = loss or {"name": "dice_ce"}
        self.criterion = build_loss(**loss)
        self.num_classes = num_classes

        self.val_metrics = SegMetrics(num_classes=num_classes, compute_hd95=True)
        self.test_metrics = SegMetrics(num_classes=num_classes, compute_hd95=True)
        self._val_viz_logged = False

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _infer(self, x: torch.Tensor) -> torch.Tensor:
        if self.hparams.sliding_window:
            return sliding_window_inference(
                x,
                roi_size=tuple(self.hparams.sw_roi_size),
                sw_batch_size=self.hparams.sw_batch_size,
                predictor=self.net,
                overlap=self.hparams.sw_overlap,
            )
        return self.net(x)

    # ------------------------------------------------------------------- steps
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images, labels = batch["image"], batch["label"]
        logits = self.net(images)
        loss = self.criterion(logits, labels)
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=images.shape[0],
        )
        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_metrics.reset()
        self._val_viz_logged = False

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        images, labels = batch["image"], batch["label"]
        logits = self._infer(images)
        loss = self.criterion(logits, labels)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, batch_size=images.shape[0])
        self.val_metrics.update(logits, labels)

        if (
            not self._val_viz_logged
            and batch_idx == 0
            and (self.current_epoch % max(1, self.hparams.viz_every_n_epochs) == 0)
        ):
            self._log_overlay(images, labels, logits, tag="val/prediction")
            self._val_viz_logged = True

    def on_validation_epoch_end(self) -> None:
        results = self.val_metrics.aggregate()
        # Monitored metric (val/dice_mean_fg) on the progress bar; the rest logged quietly.
        mean_fg = results.pop("dice_mean_fg", 0.0)
        self.log("val/dice_mean_fg", mean_fg, prog_bar=True)
        self.log_dict({f"val/{k}": v for k, v in results.items()}, prog_bar=False)

    def on_test_epoch_start(self) -> None:
        self.test_metrics.reset()

    def test_step(self, batch: dict, batch_idx: int) -> None:
        images, labels = batch["image"], batch["label"]
        logits = self._infer(images)
        self.test_metrics.update(logits, labels)

    def on_test_epoch_end(self) -> None:
        results = self.test_metrics.aggregate()
        self.log_dict({f"test/{k}": v for k, v in results.items()})

    # -------------------------------------------------------------- viz helper
    def _log_overlay(self, images, labels, logits, tag: str) -> None:
        logger = getattr(self, "logger", None)
        exp = getattr(logger, "experiment", None)
        if exp is None or not hasattr(exp, "log"):
            return
        try:
            import wandb

            pred = torch.argmax(logits[0], dim=0)
            panel = overlay_prediction(images[0], labels[0], pred)
            exp.log({tag: wandb.Image(panel, caption="image | GT | pred")})
        except Exception:  # pragma: no cover - never let logging crash training
            pass

    # -------------------------------------------------------------- optimizers
    def configure_optimizers(self):
        opt_cfg = dict(self.hparams.optimizer or {"name": "adamw", "lr": 1e-3})
        sched_cfg = dict(self.hparams.scheduler or {"name": "none"})
        name = opt_cfg.pop("name", "adamw")
        warmup = sched_cfg.pop("warmup_epochs", 0)  # noqa: F841 (reserved for future use)
        sched_name = sched_cfg.pop("name", "none")

        if name == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), **opt_cfg)
        elif name == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), momentum=0.9, nesterov=True, **opt_cfg)
        else:
            raise ValueError(f"Unknown optimizer {name!r}")

        if sched_name not in ("cosine", "polynomial"):
            return optimizer

        # self.trainer is only available once attached; fall back to a default otherwise.
        try:
            max_epochs = self.trainer.max_epochs or 100
        except RuntimeError:
            max_epochs = 100
        if sched_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        else:  # polynomial
            scheduler = torch.optim.lr_scheduler.PolynomialLR(
                optimizer, total_iters=max_epochs, power=0.9
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

"""Run inference on a single volume or a folder of volumes, writing .nrrd labels.

Example:
    python -m mvseg.predict ckpt_path=... input=data/raw/P00123_IMG04_17_volume.nrrd output=preds/
    python -m mvseg.predict ckpt_path=... input=data/raw output=preds/
"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from monai.data import MetaTensor
from monai.inferers import sliding_window_inference
from monai.transforms import SaveImage
from omegaconf import DictConfig

from mvseg.data.transforms import eval_transforms
from mvseg.models.lit_module import MVSegLitModule
from mvseg.utils.logging import configure_console, get_pylogger
from mvseg.utils.seed import set_reproducibility

configure_console()
log = get_pylogger()


def _gather_inputs(input_path: Path, image_suffix: str, file_ext: str) -> list[Path]:
    if input_path.is_dir():
        # only intensity volumes, not the _gt label files that may sit alongside
        return sorted(input_path.glob(f"*{image_suffix}{file_ext}"))
    return [input_path]


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if not cfg.get("ckpt_path"):
        raise ValueError("Provide ckpt_path=... for inference.")
    if not cfg.get("input"):
        raise ValueError("Provide input=<file-or-folder> for inference.")
    set_reproducibility(cfg.seed, cfg.deterministic)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = instantiate(cfg.model.net)
    model = MVSegLitModule.load_from_checkpoint(cfg.ckpt_path, net=net).to(device).eval()

    file_ext = cfg.data.file_ext
    tfm = eval_transforms(normalize=cfg.data.normalize)
    out_dir = Path(cfg.get("output", "preds"))
    out_dir.mkdir(parents=True, exist_ok=True)
    saver = SaveImage(
        output_dir=str(out_dir),
        output_postfix="seg",
        output_ext=file_ext,
        separate_folder=False,
        resample=False,
        print_log=False,
    )

    inputs = _gather_inputs(Path(cfg.input), cfg.data.image_suffix, file_ext)
    log.info("Predicting %d volume(s) -> %s", len(inputs), out_dir)

    for img_path in inputs:
        # eval_transforms expects both keys; feed the image as its own "label" placeholder.
        data = tfm({"image": str(img_path), "label": str(img_path)})
        image = data["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            logits = (
                sliding_window_inference(
                    image,
                    roi_size=tuple(cfg.model.sw_roi_size),
                    sw_batch_size=cfg.model.sw_batch_size,
                    predictor=model.net,
                    overlap=cfg.model.sw_overlap,
                )
                if cfg.model.sliding_window
                else model.net(image)
            )
        pred = torch.argmax(logits, dim=1, keepdim=True)[0].cpu()
        meta = getattr(data["image"], "meta", {})
        saver(MetaTensor(pred, meta=meta))
        log.info("  %s", img_path.name)


if __name__ == "__main__":
    main()

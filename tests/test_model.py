"""Model forward-pass shape tests."""

from __future__ import annotations

import torch

from mvseg.models.lit_module import MVSegLitModule
from mvseg.models.residual_unet import build_residual_unet


def test_residual_unet_output_shape(spatial_size, num_classes):
    net = build_residual_unet(out_channels=num_classes, channels=(8, 16, 32), strides=(2, 2))
    x = torch.randn(2, 1, *spatial_size)
    y = net(x)
    assert y.shape == (2, num_classes, *spatial_size)


def test_lit_module_training_step(sample_batch, num_classes):
    net = build_residual_unet(out_channels=num_classes, channels=(8, 16, 32), strides=(2, 2))
    model = MVSegLitModule(net=net, num_classes=num_classes, loss={"name": "dice_ce"})
    loss = model.training_step(sample_batch, 0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()  # gradients flow


def test_lit_module_configure_optimizers(num_classes):
    net = build_residual_unet(out_channels=num_classes, channels=(8, 16), strides=(2,))
    model = MVSegLitModule(
        net=net,
        num_classes=num_classes,
        optimizer={"name": "adamw", "lr": 1e-3},
        scheduler={"name": "none"},
    )
    opt = model.configure_optimizers()
    assert isinstance(opt, torch.optim.Optimizer)

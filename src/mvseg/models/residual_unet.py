"""3D Residual UNet factory (MONAI).

MONAI's ``UNet`` becomes a *residual* UNet when ``num_res_units > 0`` — each
conv block gains a residual skip connection. This is the standard MONAI
"ResidualUNet" configuration.
"""

from __future__ import annotations

from collections.abc import Sequence

from monai.networks.nets import UNet


def build_residual_unet(
    spatial_dims: int = 3,
    in_channels: int = 1,
    out_channels: int = 5,
    channels: Sequence[int] = (16, 32, 64, 128, 256),
    strides: Sequence[int] = (2, 2, 2, 2),
    num_res_units: int = 2,
    norm: str = "INSTANCE",
    dropout: float = 0.0,
) -> UNet:
    """Construct a 3D Residual UNet for multi-class segmentation.

    ``out_channels`` should equal the number of classes (including background);
    the network emits raw logits (apply softmax in the loss/metric).
    """
    return UNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=tuple(channels),
        strides=tuple(strides),
        num_res_units=num_res_units,
        norm=norm,
        dropout=dropout,
    )

# SPDX-License-Identifier: AGPL-3.0-or-later
"""PDFN modules reconstructed from the paper's diagrams and forward equations.

The supplied manuscript specifies the architecture and equations, but not the original
source code or every internal width. This implementation is therefore a transparent,
runnable reconstruction rather than a byte-identical copy of unpublished author code.
"""

from __future__ import annotations

import math
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ultralytics.nn.modules.conv import Conv, DWConv
    from ultralytics.nn.modules.head import Detect
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PDFN requires Ultralytics. Install the pinned environment with "
        "`pip install -r requirements.txt`."
    ) from exc


class DepthwiseSeparableConv(nn.Module):
    """Depthwise k×k convolution followed by pointwise 1×1 convolution."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 1,
        d: int = 1,
        final_act: bool = True,
    ) -> None:
        super().__init__()
        self.dw = Conv(c1, c1, k=k, s=s, g=c1, d=d)
        self.pw = Conv(c1, c2, k=1, s=1, act=final_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class RFMBlock(nn.Module):
    """Internal receptive-field modulation unit of RFM-C3k2."""

    def __init__(self, channels: int, gate_ratio: int = 4) -> None:
        super().__init__()
        if channels < 2:
            raise ValueError("RFMBlock requires at least 2 channels.")
        c_local = channels // 2
        c_context = channels - c_local
        hidden = max(8, channels // max(1, gate_ratio))

        self.c_local = c_local
        self.local = Conv(c_local, c_local, k=3, s=1)
        self.context = DepthwiseSeparableConv(c_context, c_context, k=5)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.out_proj = Conv(channels, channels, k=1, s=1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_local, x_context = torch.split(
            x, (self.c_local, x.shape[1] - self.c_local), dim=1
        )
        f_local = self.local(x_local)
        f_context = self.context(x_context)
        fused = torch.cat((f_local, f_context), dim=1)
        gate = self.gate(self.gap(x))
        return x + self.out_proj(fused * gate)


class RFMC3k2(nn.Module):
    """RFM-C3k2 with the parser-facing signature of Ultralytics C3k2."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
        gate_ratio: int = 4,
    ) -> None:
        super().__init__()
        del c3k, g, shortcut
        self.c = max(2, int(c2 * e))
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(RFMBlock(self.c, gate_ratio=gate_ratio) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class MSPC(nn.Module):
    """Multi-Scale Pyramid Convolution replacing SPPF."""

    def __init__(self, c1: int, c2: int, e: float = 0.25) -> None:
        super().__init__()
        hidden = max(8, int(c2 * e))
        self.reduce = Conv(c1, hidden, k=1, s=1)
        self.branch3 = DepthwiseSeparableConv(hidden, hidden, k=3)
        self.branch5 = DepthwiseSeparableConv(hidden, hidden, k=5)
        self.branch_dilated = Conv(hidden, hidden, k=3, s=1, d=2)
        self.project = Conv(hidden * 3, c2, k=1, s=1, act=False)
        self.shortcut = nn.Identity() if c1 == c2 else Conv(c1, c2, k=1, s=1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.reduce(x)
        z = torch.cat(
            (self.branch3(q), self.branch5(q), self.branch_dilated(q)), dim=1
        )
        return self.shortcut(x) + self.project(z)


class DFRM(nn.Module):
    """Dynamic Feature Refinement Module.

    The high-level feature is resized to the low-level spatial grid. Both streams are
    projected to ``c2`` channels, jointly generate spatial/channel responses, and are
    combined through residual fidelity plus dynamic modulation.
    """

    def __init__(
        self,
        c_low: int,
        c_high: int,
        c2: int,
        gate_ratio: int = 4,
        interpolation: str = "bilinear",
        align_groups: int = 2,
    ) -> None:
        super().__init__()
        hidden = max(8, c2 // max(1, gate_ratio))
        self.interpolation = interpolation
        low_groups = math.gcd(max(1, align_groups), math.gcd(c_low, c2))
        high_groups = math.gcd(max(1, align_groups), math.gcd(c_high, c2))
        self.low_align = Conv(c_low, c2, k=1, s=1, g=low_groups)
        self.high_align = Conv(c_high, c2, k=1, s=1, g=high_groups)
        self.spatial_response = nn.Sequential(
            nn.Conv2d(2 * c2, 1, kernel_size=1, bias=True), nn.Sigmoid()
        )
        self.channel_response = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * c2, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c2, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.out_proj = Conv(c2, c2, k=1, s=1, act=False)

    def forward(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(x) != 2:
            raise ValueError("DFRM expects [low_level_feature, high_level_feature].")
        low, high = x
        if high.shape[-2:] != low.shape[-2:]:
            if self.interpolation == "bilinear":
                high = F.interpolate(
                    high, size=low.shape[-2:], mode="bilinear", align_corners=False
                )
            else:
                high = F.interpolate(high, size=low.shape[-2:], mode=self.interpolation)

        low_a = self.low_align(low)
        high_a = self.high_align(high)
        interaction = torch.cat((low_a, high_a), dim=1)
        w_s = self.spatial_response(interaction)
        w_c = self.channel_response(interaction)
        fidelity = low_a + high_a
        dynamic = fidelity * w_s * w_c
        return self.out_proj(fidelity + dynamic)


class SACM(nn.Module):
    """Scale Alignment Calibration Module.

    A per-channel 1×1 statistical projection followed by BN and a depthwise 3×3 local
    compensation branch are residually added to the input. Grouped 1×1 projection is used
    to retain the paper's lightweight design and channel-wise statistical interpretation.
    """

    def __init__(self, c1: int, c2: int = None) -> None:
        super().__init__()
        c2 = c1 if c2 is None else c2
        if c1 != c2:
            raise ValueError("SACM is channel preserving; c1 must equal c2.")
        self.stat_conv = nn.Conv2d(c1, c1, kernel_size=1, groups=c1, bias=False)
        self.stat_bn = nn.BatchNorm2d(c1)
        self.local = DWConv(c1, c1, k=3, s=1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t_c = self.stat_bn(self.stat_conv(x))
        t_l = self.local(x)
        return x + t_c + t_l


class SADDetect(Detect):
    """Scale-Adaptive Decoupled Head compatible with Ultralytics detection loss."""

    def __init__(self, nc: int = 80, ch: Sequence[int] = ()) -> None:
        super().__init__(nc=nc, ch=tuple(ch))
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], min(self.nc, 100))

        # Lightweight regression backbone plus local position compensation.
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                DepthwiseSeparableConv(x, c2, k=3),
                DepthwiseSeparableConv(c2, c2, k=3),
            )
            for x in ch
        )
        self.reg_comp = nn.ModuleList(DWConv(c2, c2, k=3, act=False) for _ in ch)
        self.box_pred = nn.ModuleList(
            nn.Conv2d(c2, 4 * self.reg_max, kernel_size=1) for _ in ch
        )

        # YOLO11 lightweight classification backbone with scale-aware reweighting.
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
            )
            for x in ch
        )
        self.scale_mlp = nn.ModuleList()
        for x in ch:
            hidden = max(8, c3 // 16)
            self.scale_mlp.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(x, hidden, kernel_size=1, bias=True),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(hidden, c3, kernel_size=1, bias=True),
                    nn.Sigmoid(),
                )
            )
        self.cls_pred = nn.ModuleList(
            nn.Conv2d(c3, self.nc, kernel_size=1) for _ in ch
        )

    def forward(self, x: List[torch.Tensor]):
        if getattr(self, "end2end", False):
            raise NotImplementedError("SAD-Head targets the standard one-to-many YOLO11 detection path.")
        for i in range(self.nl):
            base_reg = self.cv2[i](x[i])
            enhanced_reg = base_reg + self.reg_comp[i](base_reg)
            box = self.box_pred[i](enhanced_reg)

            base_cls = self.cv3[i](x[i])
            scale = self.scale_mlp[i](x[i])
            cls = self.cls_pred[i](base_cls * scale)
            x[i] = torch.cat((box, cls), dim=1)

        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def bias_init(self) -> None:
        for box, cls, stride in zip(self.box_pred, self.cls_pred, self.stride):
            box.bias.data[:] = 1.0
            cls.bias.data[: self.nc] = math.log(5 / self.nc / (640 / stride) ** 2)


__all__ = (
    "DepthwiseSeparableConv",
    "RFMBlock",
    "RFMC3k2",
    "MSPC",
    "DFRM",
    "SACM",
    "SADDetect",
)

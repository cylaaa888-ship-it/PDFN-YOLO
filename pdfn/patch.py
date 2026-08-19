# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ultralytics parser registration for PDFN custom modules.

The parser below is adapted from Ultralytics 8.3.59 ``nn/tasks.py::parse_model``
(AGPL-3.0) and adds explicit channel handling for RFMC3k2, MSPC, DFRM, SACM,
and SADDetect. No installed package files are modified.
"""

from __future__ import annotations

import contextlib
import os
import warnings

import torch
import torch.nn as nn

from .modules import DFRM, MSPC, RFMC3k2, SACM, SADDetect

_PATCHED = False
_SUPPORTED_VERSION = "8.3.59"


def _make_parser(tasks):
    def parse_model_pdfn(d, ch, verbose=True):
        import ast

        legacy = True
        max_channels = float("inf")
        nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
        depth, width, kpt_shape = (
            d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape")
        )
        scale = d.get("scale", "")
        if scales:
            if not scale:
                scale = tuple(scales.keys())[0]
                tasks.LOGGER.warning(
                    f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'."
                )
            depth, width, max_channels = scales[scale]
        if act:
            tasks.Conv.default_act = eval(act)
            if verbose:
                tasks.LOGGER.info(f"{tasks.colorstr('activation:')} {act}")
        if verbose:
            tasks.LOGGER.info(
                f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  "
                f"{'module':<45}{'arguments':<30}"
            )

        ch = [ch]
        layers, save, c2 = [], [], ch[-1]
        base_modules = {
            tasks.Classify,
            tasks.Conv,
            tasks.ConvTranspose,
            tasks.GhostConv,
            tasks.Bottleneck,
            tasks.GhostBottleneck,
            tasks.SPP,
            tasks.SPPF,
            tasks.C2fPSA,
            tasks.C2PSA,
            tasks.DWConv,
            tasks.Focus,
            tasks.BottleneckCSP,
            tasks.C1,
            tasks.C2,
            tasks.C2f,
            tasks.C3k2,
            tasks.RepNCSPELAN4,
            tasks.ELAN1,
            tasks.ADown,
            tasks.AConv,
            tasks.SPPELAN,
            tasks.C2fAttn,
            tasks.C3,
            tasks.C3TR,
            tasks.C3Ghost,
            nn.ConvTranspose2d,
            tasks.DWConvTranspose2d,
            tasks.C3x,
            tasks.RepC3,
            tasks.PSA,
            tasks.SCDown,
            tasks.C2fCIB,
            RFMC3k2,
            MSPC,
        }
        repeat_modules = {
            tasks.BottleneckCSP,
            tasks.C1,
            tasks.C2,
            tasks.C2f,
            tasks.C3k2,
            tasks.C2fAttn,
            tasks.C3,
            tasks.C3TR,
            tasks.C3Ghost,
            tasks.C3x,
            tasks.RepC3,
            tasks.C2fPSA,
            tasks.C2fCIB,
            tasks.C2PSA,
            RFMC3k2,
        }

        for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
            m = getattr(torch.nn, m[3:]) if "nn." in m else getattr(tasks, m)
            for j, a in enumerate(args):
                if isinstance(a, str):
                    with contextlib.suppress(ValueError):
                        args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
            n = n_ = max(round(n * depth), 1) if n > 1 else n

            if m in base_modules:
                c1, c2 = ch[f], args[0]
                if c2 != nc:
                    c2 = tasks.make_divisible(min(c2, max_channels) * width, 8)
                if m is tasks.C2fAttn:
                    args[1] = tasks.make_divisible(
                        min(args[1], max_channels // 2) * width, 8
                    )
                    args[2] = int(
                        max(round(min(args[2], max_channels // 2 // 32)) * width, 1)
                        if args[2] > 1
                        else args[2]
                    )
                args = [c1, c2, *args[1:]]
                if m in repeat_modules:
                    args.insert(2, n)
                    n = 1
                if m in {tasks.C3k2, RFMC3k2}:
                    legacy = False
                    if scale in "mlx" and len(args) > 3:
                        args[3] = True

            elif m is DFRM:
                if not isinstance(f, list) or len(f) != 2:
                    raise ValueError("DFRM YAML 'from' must contain exactly two layer indices.")
                c_low, c_high = ch[f[0]], ch[f[1]]
                c2 = args[0]
                if c2 != nc:
                    c2 = tasks.make_divisible(min(c2, max_channels) * width, 8)
                args = [c_low, c_high, c2, *args[1:]]

            elif m is SACM:
                c1 = ch[f]
                c2 = c1
                args = [c1, c2, *args]

            elif m is tasks.AIFI:
                args = [ch[f], *args]
            elif m in {tasks.HGStem, tasks.HGBlock}:
                c1, cm, c2 = ch[f], args[0], args[1]
                args = [c1, cm, c2, *args[2:]]
                if m is tasks.HGBlock:
                    args.insert(4, n)
                    n = 1
            elif m is tasks.ResNetLayer:
                c2 = args[1] if args[3] else args[1] * 4
            elif m is nn.BatchNorm2d:
                args = [ch[f]]
            elif m is tasks.Concat:
                c2 = sum(ch[x] for x in f)
            elif m in {
                tasks.Detect,
                tasks.WorldDetect,
                tasks.Segment,
                tasks.Pose,
                tasks.OBB,
                tasks.ImagePoolingAttn,
                tasks.v10Detect,
                SADDetect,
            }:
                args.append([ch[x] for x in f])
                if m is tasks.Segment:
                    args[2] = tasks.make_divisible(
                        min(args[2], max_channels) * width, 8
                    )
                if m in {tasks.Detect, tasks.Segment, tasks.Pose, tasks.OBB, SADDetect}:
                    m.legacy = legacy
            elif m is tasks.RTDETRDecoder:
                args.insert(1, [ch[x] for x in f])
            elif m in {tasks.CBLinear, tasks.TorchVision, tasks.Index}:
                c2 = args[0]
                c1 = ch[f]
                args = [c1, c2, *args[1:]]
            elif m is tasks.CBFuse:
                c2 = ch[f[-1]]
            else:
                c2 = ch[f]

            m_ = (
                nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
            )
            t = str(m)[8:-2].replace("__main__.", "")
            m_.np = sum(x.numel() for x in m_.parameters())
            m_.i, m_.f, m_.type = i, f, t
            if verbose:
                tasks.LOGGER.info(
                    f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  "
                    f"{t:<45}{str(args):<30}"
                )
            save.extend(
                x % i for x in ([f] if isinstance(f, int) else f) if x != -1
            )
            layers.append(m_)
            if i == 0:
                ch = []
            ch.append(c2)
        return nn.Sequential(*layers), sorted(save)

    return parse_model_pdfn


def register_pdfn_modules(strict_version: bool = True) -> None:
    """Register PDFN classes and the custom parser before constructing a YOLO model."""
    global _PATCHED
    if _PATCHED:
        return

    import ultralytics
    import ultralytics.nn.tasks as tasks

    version = getattr(ultralytics, "__version__", "unknown")
    allow_unsupported = os.getenv("PDFN_ALLOW_UNSUPPORTED_ULTRALYTICS", "0") == "1"
    if version != _SUPPORTED_VERSION:
        message = (
            f"PDFN was validated against ultralytics=={_SUPPORTED_VERSION}, "
            f"but found {version}."
        )
        if strict_version and not allow_unsupported:
            raise RuntimeError(
                message
                + " Install requirements.txt or set PDFN_ALLOW_UNSUPPORTED_ULTRALYTICS=1 "
                "after checking compatibility."
            )
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    tasks.RFMC3k2 = RFMC3k2
    tasks.MSPC = MSPC
    tasks.DFRM = DFRM
    tasks.SACM = SACM
    tasks.SADDetect = SADDetect
    tasks.parse_model = _make_parser(tasks)
    _PATCHED = True

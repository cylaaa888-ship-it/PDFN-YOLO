#!/usr/bin/env python3
"""Measure forward-only FPS (batch=1), excluding preprocessing and NMS."""

from __future__ import annotations

import argparse
import time

import torch

import _bootstrap  # noqa: F401  # adds repository root to sys.path



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()

    from pdfn.patch import register_pdfn_modules
    register_pdfn_modules()
    from ultralytics import YOLO

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for a GPU-comparable FPS measurement.")

    device = torch.device(args.device)
    wrapper = YOLO(args.weights)
    model = wrapper.model.to(device).eval()
    x = torch.randn(1, 3, 640, 640, device=device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(args.iterations):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

    latency_ms = 1000.0 * elapsed / args.iterations
    fps = args.iterations / elapsed
    print(f"forward latency: {latency_ms:.3f} ms/image")
    print(f"forward-only FPS: {fps:.2f}")


if __name__ == "__main__":
    main()

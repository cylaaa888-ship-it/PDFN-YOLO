from pathlib import Path

import pytest

pytest.importorskip("ultralytics", reason="runtime module tests require the pinned Ultralytics dependency")
import torch

from pdfn.modules import DFRM, MSPC, RFMC3k2, SACM
from pdfn.model import build_pdfn


def test_rfm_c3k2_shape():
    module = RFMC3k2(64, 128, n=2, e=0.5)
    assert module(torch.randn(2, 64, 80, 80)).shape == (2, 128, 80, 80)


def test_mspc_shape_and_projection_shortcut():
    module = MSPC(96, 128)
    assert module(torch.randn(2, 96, 20, 20)).shape == (2, 128, 20, 20)


def test_dfrm_shape_resize_and_input_validation():
    module = DFRM(64, 128, 96)
    output = module([torch.randn(2, 64, 40, 40), torch.randn(2, 128, 20, 20)])
    assert output.shape == (2, 96, 40, 40)
    with pytest.raises(ValueError):
        module([torch.randn(1, 64, 10, 10)])


def test_sacm_shape_and_channel_guard():
    module = SACM(96, 96)
    x = torch.randn(2, 96, 40, 40)
    assert module(x).shape == x.shape
    with pytest.raises(ValueError):
        SACM(96, 64)


def test_full_model_build_and_forward():
    root = Path(__file__).resolve().parents[1]
    model = build_pdfn(root / "configs/yolo11s-pdfn-v2.yaml", pretrained=None)
    model.model.train()
    output = model.model(torch.randn(1, 3, 640, 640))
    assert isinstance(output, list) and len(output) == 3

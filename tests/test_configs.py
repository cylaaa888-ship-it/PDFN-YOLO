from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ABLATIONS = ROOT / "configs" / "ablations"

EXPECTED = {
    "a_baseline.yaml": (0, 0, 0, 0, 0),
    "b_rfm_c3k2.yaml": (4, 0, 0, 0, 0),
    "c_mspc.yaml": (0, 1, 0, 0, 0),
    "d_backbone.yaml": (4, 1, 0, 0, 0),
    "e_dfrm.yaml": (0, 0, 4, 0, 0),
    "f_sacm.yaml": (0, 0, 0, 4, 0),
    "g_neck.yaml": (0, 0, 4, 4, 0),
    "h_sad_head.yaml": (0, 0, 0, 0, 1),
    "i_backbone_neck.yaml": (4, 1, 4, 4, 0),
    "j_neck_sad_head.yaml": (0, 0, 4, 4, 1),
    "k_pdfn_full.yaml": (4, 1, 4, 4, 1),
}


def load(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def modules(cfg):
    return [layer[2] for layer in cfg["backbone"] + cfg["head"]]


def validate_refs(cfg):
    layers = cfg["backbone"] + cfg["head"]
    for i, layer in enumerate(layers):
        source = layer[0]
        refs = source if isinstance(source, list) else [source]
        for ref in refs:
            if ref == -1:
                continue
            assert isinstance(ref, int)
            if ref >= 0:
                assert ref < i, f"layer {i} references future/nonexistent layer {ref}"
            else:
                assert i + ref >= 0, f"layer {i} has out-of-range relative ref {ref}"


def test_all_ablation_configs_are_minimal_and_well_referenced():
    assert {p.name for p in ABLATIONS.glob("*.yaml")} == set(EXPECTED)
    for path in sorted(ABLATIONS.glob("*.yaml")):
        cfg = load(path)
        mods = modules(cfg)
        assert "nn.Identity" not in mods, f"no-op placeholder remains in {path.name}"
        validate_refs(cfg)
        expected = EXPECTED[path.name]
        actual = tuple(mods.count(x) for x in ("RFMC3k2", "MSPC", "DFRM", "SACM", "SADDetect"))
        assert actual == expected, (path.name, actual, expected)


def test_baseline_matches_canonical_yolo11_head_indexing():
    cfg = load(ABLATIONS / "a_baseline.yaml")
    assert len(cfg["backbone"]) == 11
    assert len(cfg["head"]) == 13
    assert cfg["backbone"][9][2] == "SPPF"
    assert cfg["head"][-1] == [[16, 19, 22], 1, "Detect", ["nc"]]


def test_full_yaml_matches_full_ablation_topology():
    main = load(ROOT / "configs" / "yolo11s-pdfn-v2.yaml")
    ablation = load(ABLATIONS / "k_pdfn_full.yaml")
    assert main == ablation
    validate_refs(main)

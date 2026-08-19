#!/usr/bin/env python3
"""Summarize val box/cls/DFL losses over the final N epochs of Ultralytics results.csv."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

LOSS_KEYS=("val/box_loss","val/cls_loss","val/dfl_loss")


def load_rows(path: Path):
    with path.open(encoding="utf-8-sig") as f:
        raw=list(csv.DictReader(f))
    return [{(k or "").strip():(v or "").strip() for k,v in row.items()} for row in raw]


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--input", action="append", required=True, metavar="MODEL=RESULTS.CSV")
    p.add_argument("--last", type=int, default=20)
    p.add_argument("--output", type=Path, default=None)
    a=p.parse_args()
    if a.last < 2:
        raise ValueError("--last must be >=2 to compute a sample SD")
    out=[]
    for spec in a.input:
        if "=" not in spec:
            raise ValueError(f"expected MODEL=RESULTS.CSV, got {spec}")
        model, path=spec.split("=",1)
        rows=load_rows(Path(path))
        if len(rows)<a.last:
            raise ValueError(f"{path}: only {len(rows)} epochs, need at least {a.last}")
        rows=rows[-a.last:]
        item={"model":model,"window_epochs":a.last}
        for key in LOSS_KEYS:
            if key not in rows[0]:
                raise ValueError(f"{path}: missing column {key}")
            vals=[float(r[key]) for r in rows]
            short=key.replace("val/","")
            item[f"{short}_mean"]=statistics.mean(vals)
            item[f"{short}_sd"]=statistics.stdev(vals)
        out.append(item)
    fields=["model","window_epochs","box_loss_mean","box_loss_sd","cls_loss_mean","cls_loss_sd","dfl_loss_mean","dfl_loss_sd"]
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(out)
    for r in out:
        print(f"{r['model']}: box={r['box_loss_mean']:.4f}±{r['box_loss_sd']:.4f}, "
              f"cls={r['cls_loss_mean']:.4f}±{r['cls_loss_sd']:.4f}, "
              f"dfl={r['dfl_loss_mean']:.4f}±{r['dfl_loss_sd']:.4f}")


if __name__ == "__main__":
    main()

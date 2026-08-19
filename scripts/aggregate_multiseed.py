#!/usr/bin/env python3
"""Aggregate measured multi-seed metrics as mean ± sample standard deviation."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

REQUIRED = ("model", "seed", "map50", "map50_95", "apsmall")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="CSV with model,seed,map50,map50_95,apsmall")
    p.add_argument("--output", type=Path, default=None)
    a = p.parse_args()
    rows = list(csv.DictReader(a.input.open(encoding="utf-8-sig")))
    if not rows:
        raise ValueError("input CSV is empty")
    missing = [x for x in REQUIRED if x not in rows[0]]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["model"]].append(r)
    output=[]
    for model, items in grouped.items():
        seeds=[int(x["seed"]) for x in items]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate seed for {model}: {seeds}")
        if len(items) < 2:
            raise ValueError(f"at least two seeds are required for SD: {model}")
        out={"model":model,"n":len(items),"seeds":";".join(map(str,sorted(seeds)))}
        for key in ("map50","map50_95","apsmall"):
            vals=[float(x[key]) for x in items]
            out[f"{key}_mean"]=f"{statistics.mean(vals):.4f}"
            out[f"{key}_sd"]=f"{statistics.stdev(vals):.4f}"
        output.append(out)
    fields=["model","n","seeds","map50_mean","map50_sd","map50_95_mean","map50_95_sd","apsmall_mean","apsmall_sd"]
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        with a.output.open("w", newline="", encoding="utf-8") as f:
            w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(output)
    for r in output:
        print(f"{r['model']}: mAP50={r['map50_mean']}±{r['map50_sd']}, "
              f"mAP50-95={r['map50_95_mean']}±{r['map50_95_sd']}, "
              f"APsmall={r['apsmall_mean']}±{r['apsmall_sd']} (n={r['n']})")


if __name__ == "__main__":
    main()

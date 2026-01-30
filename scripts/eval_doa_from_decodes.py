import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

AZ_RE = re.compile(r"azimuth[^\d+-]*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE)
EL_RE = re.compile(r"elevation[^\d+-]*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE)
NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def parse_az_el(text: str) -> Tuple[Optional[float], Optional[float]]:
    if not text:
        return None, None
    az_m = AZ_RE.search(text)
    el_m = EL_RE.search(text)
    az = float(az_m.group(1)) if az_m else None
    el = float(el_m.group(1)) if el_m else None
    if az is not None and el is not None:
        return az, el

    nums = NUM_RE.findall(text)
    if len(nums) >= 2:
        try:
            return float(nums[0]), float(nums[1])
        except Exception:
            return None, None
    return None, None


def wrap_azimuth_diff(pred: float, gt: float) -> float:
    diff = pred - gt
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff


def angular_error_norm(az1, el1, az2, el2) -> Optional[float]:
    if None in (az1, el1, az2, el2):
        return None
    try:
        az1_r = math.radians(az1)
        az2_r = math.radians(az2)
        el1_r = math.radians(el1)
        el2_r = math.radians(el2)

        x1 = math.cos(el1_r) * math.cos(az1_r)
        y1 = math.cos(el1_r) * math.sin(az1_r)
        z1 = math.sin(el1_r)

        x2 = math.cos(el2_r) * math.cos(az2_r)
        y2 = math.cos(el2_r) * math.sin(az2_r)
        z2 = math.sin(el2_r)

        dot = x1 * x2 + y1 * y2 + z1 * z2
        dot = max(min(dot, 1.0), -1.0)
        return math.degrees(math.acos(dot))
    except Exception:
        return None


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan"), "count": 0}
    arr = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(arr.mean().item()),
        "median": float(arr.median().item()),
        "std": float(arr.std(unbiased=False).item()),
        "count": int(arr.numel()),
    }


def load_kv(path: Path) -> Dict[str, str]:
    data = {}
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if "\t" in line:
                key, text = line.split("\t", 1)
            else:
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    key, text = parts
                else:
                    continue
            data[key] = text
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="decode log pred file (key\\ttext)")
    parser.add_argument("--gt", required=True, help="decode log gt file (key\\ttext)")
    parser.add_argument("--out-metrics", required=True, help="output metrics json")
    parser.add_argument("--out-details", required=True, help="output per-sample json")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--filter-doa", action="store_true")
    args = parser.parse_args()

    pred_map = load_kv(Path(args.pred))
    gt_map = load_kv(Path(args.gt))

    az_errs = []
    el_errs = []
    ang_errs = []
    details = []
    seen = 0

    for key, gt_text in gt_map.items():
        pred_text = pred_map.get(key, "")
        gt_az, gt_el = parse_az_el(gt_text)
        pred_az, pred_el = parse_az_el(pred_text)

        if args.filter_doa and (gt_az is None or gt_el is None):
            continue
        if pred_az is None or pred_el is None or gt_az is None or gt_el is None:
            details.append({
                "key": key,
                "prediction": pred_text,
                "target": gt_text,
                "pred_az": pred_az,
                "pred_el": pred_el,
                "gt_az": gt_az,
                "gt_el": gt_el,
                "az_error": None,
                "el_error": None,
                "angular_error": None,
            })
            continue

        az_err = abs(wrap_azimuth_diff(pred_az, gt_az))
        el_err = abs(pred_el - gt_el)
        ang_err = angular_error_norm(pred_az, pred_el, gt_az, gt_el)

        az_errs.append(az_err)
        el_errs.append(el_err)
        if ang_err is not None:
            ang_errs.append(ang_err)

        details.append({
            "key": key,
            "prediction": pred_text,
            "target": gt_text,
            "pred_az": pred_az,
            "pred_el": pred_el,
            "gt_az": gt_az,
            "gt_el": gt_el,
            "az_error": az_err,
            "el_error": el_err,
            "angular_error": ang_err,
        })

        seen += 1
        if args.max_samples and seen >= args.max_samples:
            break

    metrics = {
        "az_error": summarize(az_errs),
        "el_error": summarize(el_errs),
        "angular_error": summarize(ang_errs),
        "total_pred": len(pred_map),
        "total_gt": len(gt_map),
        "paired": len(details),
    }

    Path(args.out_metrics).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_metrics, "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    Path(args.out_details).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_details, "w") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

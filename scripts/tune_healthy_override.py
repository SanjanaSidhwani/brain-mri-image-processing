import argparse
import gzip
import json
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.aggregation.topk_aggregation import robust_patient_prediction_from_tumor_probs
from src.dataset.input_transforms import build_eval_transform
from src.dataset.mri_dataset import MRISliceDataset
from src.evaluation.predictor import Predictor


def parse_args():
    parser = argparse.ArgumentParser(description="Tune healthy override thresholds using IXI + BraTS2021")
    parser.add_argument("--ixi_dataset_path", type=str, default="data/processed/ixi_t1_t2_t1ce_records.pkl.gz")
    parser.add_argument("--full_dataset_path", type=str, default="data/dataset_records.pkl.gz")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--target_size", type=int, default=224)
    parser.add_argument("--center_crop_size", type=int, default=180)
    parser.add_argument("--max_patients_per_dataset", type=int, default=300)
    parser.add_argument("--min_tumor_recall", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", type=str, default="outputs/calibration/healthy_override_tuning.json")
    parser.add_argument("--aggregation_calibration_path", type=str, default="outputs/calibration/aggregation_calibration.json")
    return parser.parse_args()


def load_gz_pickle(path: Path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def get_latest_checkpoint(checkpoint_dir: Path) -> Path:
    candidates = sorted(checkpoint_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No checkpoints in {checkpoint_dir}")
    return candidates[-1]


def choose_patients(records, max_patients, seed):
    if max_patients <= 0:
        return records

    by_patient = defaultdict(list)
    for r in records:
        by_patient[str(r["patient_id"])].append(r)

    pids = sorted(by_patient.keys())
    if len(pids) <= max_patients:
        return records

    rng = random.Random(seed)
    chosen = set(rng.sample(pids, max_patients))
    return [r for r in records if str(r["patient_id"]) in chosen]


def collect_patient_probs(records, predictor, batch_size, target_size, center_crop_size):
    transform = build_eval_transform(target_size=target_size, center_crop_size=center_crop_size)
    dataset = MRISliceDataset(
        records,
        target_size=target_size,
        transform=transform,
        channel_mode="2.5d",
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    outputs = predictor.collect_predictions(loader)

    patient_probs = defaultdict(list)
    patient_labels = {}
    for rec, prob in zip(records, outputs["probabilities"]):
        pid = str(rec["patient_id"])
        patient_probs[pid].append(float(prob[1]))
        patient_labels[pid] = int(rec["label"])

    return patient_probs, patient_labels


def evaluate_config(patient_probs, patient_labels, base_params, cfg):
    tp = tn = fp = fn = 0

    for pid, probs in patient_probs.items():
        true_label = int(patient_labels[pid])
        decision = robust_patient_prediction_from_tumor_probs(
            tumor_probs=probs,
            threshold=base_params["threshold"],
            top_k=base_params["top_k"],
            method=base_params["method"],
            min_suspicious_slices=base_params["min_suspicious_slices"],
            suspicious_prob_threshold=base_params["suspicious_prob_threshold"],
            min_suspicious_fraction=base_params["min_suspicious_fraction"],
            healthy_override_topk_max=cfg["healthy_override_topk_max"],
            healthy_override_max_suspicious_slices=cfg["healthy_override_max_suspicious_slices"],
            healthy_override_max_suspicious_fraction=cfg["healthy_override_max_suspicious_fraction"],
            hard_tumor_topk_min=cfg["hard_tumor_topk_min"],
            hard_tumor_min_suspicious_slices=cfg["hard_tumor_min_suspicious_slices"],
        )
        pred = int(decision["prediction"])

        if true_label == 1 and pred == 1:
            tp += 1
        elif true_label == 1 and pred == 0:
            fn += 1
        elif true_label == 0 and pred == 1:
            fp += 1
        else:
            tn += 1

    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    balanced_accuracy = 0.5 * (sensitivity + specificity)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
    }


def load_base_aggregation_params(path: Path):
    defaults = {
        "threshold": 0.70,
        "top_k": 20,
        "method": "median",
        "min_suspicious_slices": 8,
        "suspicious_prob_threshold": 0.90,
        "min_suspicious_fraction": 0.30,
    }
    if not path.exists():
        return defaults

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        params = payload.get("best", {}).get("params", {})
        merged = defaults.copy()
        merged.update({k: params[k] for k in defaults.keys() if k in params})
        return merged
    except Exception:
        return defaults


def apply_best_to_aggregation_calibration(path: Path, best_cfg):
    payload = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            payload = {}

    payload.setdefault("best", {})
    payload["best"].setdefault("params", {})
    payload["best"]["params"].update(best_cfg)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    args = parse_args()
    random.seed(args.seed)

    ixi_records = load_gz_pickle(Path(args.ixi_dataset_path))
    full_records = load_gz_pickle(Path(args.full_dataset_path))

    brats2021_records = [
        r for r in full_records
        if str(r.get("dataset", "")).lower() == "brats"
        and "brats2021" in str(r.get("volume_path", "")).lower()
    ]

    ixi_records = choose_patients(ixi_records, args.max_patients_per_dataset, args.seed)
    brats2021_records = choose_patients(brats2021_records, args.max_patients_per_dataset, args.seed)

    print("IXI records:", len(ixi_records))
    print("BraTS2021 records:", len(brats2021_records))
    print("IXI patients:", len({r['patient_id'] for r in ixi_records}))
    print("BraTS2021 patients:", len({r['patient_id'] for r in brats2021_records}))

    ckpt = Path(args.checkpoint_path) if args.checkpoint_path else get_latest_checkpoint(PROJECT_ROOT / "outputs" / "checkpoints")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Checkpoint:", ckpt)
    print("Device:", device)

    predictor = Predictor.load_from_checkpoint(str(ckpt), device)

    ixi_probs, ixi_labels = collect_patient_probs(
        ixi_records, predictor, args.batch_size, args.target_size, args.center_crop_size
    )
    brats_probs, brats_labels = collect_patient_probs(
        brats2021_records, predictor, args.batch_size, args.target_size, args.center_crop_size
    )

    patient_probs = {}
    patient_labels = {}
    for pid, probs in ixi_probs.items():
        patient_probs[f"ixi::{pid}"] = probs
        patient_labels[f"ixi::{pid}"] = ixi_labels[pid]
    for pid, probs in brats_probs.items():
        patient_probs[f"brats::{pid}"] = probs
        patient_labels[f"brats::{pid}"] = brats_labels[pid]

    base_params = load_base_aggregation_params(Path(args.aggregation_calibration_path))

    healthy_topk_vals = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    healthy_slice_vals = [0, 1, 2, 3, 4, 5]
    healthy_frac_vals = [0.00, 0.01, 0.02, 0.05, 0.08, 0.10]
    hard_topk_vals = [0.90, 0.93, 0.95, 0.97]
    hard_slice_vals = [1, 2, 3, 4]

    all_rows = []
    for a in healthy_topk_vals:
        for b in healthy_slice_vals:
            for c in healthy_frac_vals:
                for d in hard_topk_vals:
                    for e in hard_slice_vals:
                        cfg = {
                            "healthy_override_topk_max": float(a),
                            "healthy_override_max_suspicious_slices": int(b),
                            "healthy_override_max_suspicious_fraction": float(c),
                            "hard_tumor_topk_min": float(d),
                            "hard_tumor_min_suspicious_slices": int(e),
                        }
                        m = evaluate_config(patient_probs, patient_labels, base_params, cfg)
                        row = {**cfg, **m}
                        row["meets_min_recall"] = bool(m["sensitivity"] >= args.min_tumor_recall)
                        all_rows.append(row)

    constrained = [r for r in all_rows if r["meets_min_recall"]]
    search_space = constrained if constrained else all_rows

    best = max(
        search_space,
        key=lambda r: (r["specificity"], r["balanced_accuracy"], r["sensitivity"]),
    )

    best_cfg = {
        "healthy_override_topk_max": best["healthy_override_topk_max"],
        "healthy_override_max_suspicious_slices": best["healthy_override_max_suspicious_slices"],
        "healthy_override_max_suspicious_fraction": best["healthy_override_max_suspicious_fraction"],
        "hard_tumor_topk_min": best["hard_tumor_topk_min"],
        "hard_tumor_min_suspicious_slices": best["hard_tumor_min_suspicious_slices"],
    }

    apply_best_to_aggregation_calibration(Path(args.aggregation_calibration_path), best_cfg)

    ranked = sorted(
        all_rows,
        key=lambda r: (r["meets_min_recall"], r["specificity"], r["balanced_accuracy"], r["sensitivity"]),
        reverse=True,
    )[:20]

    out = {
        "checkpoint": str(ckpt),
        "base_params": base_params,
        "min_tumor_recall_constraint": args.min_tumor_recall,
        "num_patient_cases": len(patient_probs),
        "num_ixi_patients": len(ixi_probs),
        "num_brats2021_patients": len(brats_probs),
        "best": best,
        "best_params_applied_to_aggregation_calibration": best_cfg,
        "top_candidates": ranked,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Best tuned params:")
    print(best_cfg)
    print("Best metrics:")
    print({k: best[k] for k in ["sensitivity", "specificity", "balanced_accuracy", "tp", "tn", "fp", "fn", "meets_min_recall"]})
    print("Saved tuning report:", output_path)


if __name__ == "__main__":
    main()

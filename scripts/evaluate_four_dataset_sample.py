import argparse
import gzip
import json
import pickle
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import nibabel as nib
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.input_transforms import build_eval_transform
from src.dataset.mri_dataset import MRISliceDataset
from src.evaluation.predictor import Predictor
from src.evaluation.metrics import compute_classification_metrics
from src.aggregation.topk_aggregation import topk_patient_prediction


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate 100-sample slices from each of 4 datasets")
    parser.add_argument("--dataset_path", type=str, default="data/dataset_records.pkl.gz")
    parser.add_argument("--samples_per_dataset", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument(
        "--aggregation_config",
        type=str,
        default="outputs/calibration/aggregation_calibration.json",
        help="JSON file containing calibrated patient-level aggregation parameters.",
    )
    parser.add_argument("--output_json", type=str, default="outputs/reports/four_dataset_100_each_report.json")
    return parser.parse_args()


def load_records(path: Path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def load_aggregation_params(config_path: Path):
    defaults = {
        "threshold": 0.70,
        "top_k": 20,
        "method": "median",
        "min_suspicious_slices": 8,
        "suspicious_prob_threshold": 0.90,
        "min_suspicious_fraction": 0.30,
        "healthy_override_topk_max": 0.20,
        "healthy_override_max_suspicious_slices": 3,
        "healthy_override_max_suspicious_fraction": 0.05,
        "hard_tumor_topk_min": 0.95,
        "hard_tumor_min_suspicious_slices": 3,
    }

    if not config_path.exists():
        print(f"Aggregation config not found: {config_path}. Using defaults.")
        return defaults

    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    params = raw
    if isinstance(raw.get("best"), dict) and isinstance(raw["best"].get("params"), dict):
        params = raw["best"]["params"]
    elif isinstance(raw.get("best_params_applied_to_aggregation_calibration"), dict):
        params = raw["best_params_applied_to_aggregation_calibration"]

    merged = dict(defaults)
    for key in defaults:
        if key in params:
            merged[key] = params[key]

    return merged


def latest_checkpoint(checkpoint_dir: Path) -> Path:
    cands = sorted(checkpoint_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f"No checkpoints in {checkpoint_dir}")
    return cands[-1]


def fix_volume_path(record):
    vp = str(record.get("volume_path", ""))
    if vp and Path(vp).exists():
        return record

    if vp.endswith(".nii.gz"):
        alt = vp[:-3]
        if Path(alt).exists():
            record = dict(record)
            record["volume_path"] = alt
            return record

    return None


def is_readable_nifti(path_str: str) -> bool:
    try:
        nib.load(path_str)
        return True
    except Exception:
        return False


def pick_group_records(all_records, group_key, n, seed):
    if group_key == "brats2020":
        candidates = [
            r for r in all_records
            if str(r.get("dataset", "")).lower() == "brats"
            and "brats2020" in str(r.get("volume_path", "")).lower()
        ]
    elif group_key == "brats2021":
        candidates = [
            r for r in all_records
            if str(r.get("dataset", "")).lower() == "brats"
            and "brats2021" in str(r.get("volume_path", "")).lower()
        ]
    elif group_key == "ixi":
        candidates = [r for r in all_records if str(r.get("dataset", "")).lower() == "ixi"]
    elif group_key == "oasis":
        candidates = [r for r in all_records if str(r.get("dataset", "")).lower() == "oasis"]
    else:
        raise ValueError(f"Unknown group: {group_key}")

    rng = random.Random(seed)
    rng.shuffle(candidates)

    chosen = []
    for rec in candidates:
        fixed = fix_volume_path(rec)
        if fixed is None:
            continue
        if not is_readable_nifti(str(fixed.get("volume_path", ""))):
            continue
        chosen.append(fixed)
        if len(chosen) >= n:
            break

    if len(chosen) < n:
        raise ValueError(f"Could not collect {n} valid records for {group_key}; got {len(chosen)}")

    return chosen


def main():
    args = parse_args()
    random.seed(args.seed)

    records = load_records(Path(args.dataset_path))
    agg_params = load_aggregation_params(Path(args.aggregation_config))

    groups = ["brats2020", "brats2021", "ixi", "oasis"]
    grouped_records = {}
    for idx, g in enumerate(groups):
        grouped_records[g] = pick_group_records(records, g, args.samples_per_dataset, args.seed + idx)

    eval_records = []
    record_group = []
    for g in groups:
        eval_records.extend(grouped_records[g])
        record_group.extend([g] * len(grouped_records[g]))

    print("Sample counts by group:")
    for g in groups:
        print(g, len(grouped_records[g]))

    ckpt = Path(args.checkpoint_path) if args.checkpoint_path else latest_checkpoint(PROJECT_ROOT / "outputs" / "checkpoints")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = build_eval_transform(target_size=224, center_crop_size=180)
    dataset = MRISliceDataset(eval_records, target_size=224, transform=transform, channel_mode="2.5d")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    predictor = Predictor.load_from_checkpoint(str(ckpt), device)
    outputs = predictor.collect_predictions(loader)

    overall = compute_classification_metrics(
        outputs["true_labels"],
        outputs["predicted_labels"],
        outputs["probabilities"],
    )

    per_group = {}
    for g in groups:
        idxs = [i for i, tag in enumerate(record_group) if tag == g]
        y_true = [outputs["true_labels"][i] for i in idxs]
        y_pred = [outputs["predicted_labels"][i] for i in idxs]
        probs = [outputs["probabilities"][i] for i in idxs]

        # Some groups are single-class by design; skip ROC in that case.
        try:
            metrics = compute_classification_metrics(y_true, y_pred, probs)
            roc = metrics.get("roc_auc")
        except Exception:
            metrics = compute_classification_metrics(y_true, y_pred, None)
            roc = None
            metrics["roc_auc"] = roc

        per_group[g] = {
            "count": len(idxs),
            "label_counts": dict(Counter(y_true)),
            "prediction_counts": dict(Counter(y_pred)),
            "metrics": {
                "accuracy": float(metrics["accuracy"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1_score": float(metrics["f1_score"]),
                "sensitivity": float(metrics["sensitivity"]),
                "specificity": float(metrics["specificity"]),
                "confusion_matrix": metrics["confusion_matrix"].tolist(),
                "roc_auc": None if metrics.get("roc_auc") is None else float(metrics["roc_auc"]),
            },
        }

    patient_preds = topk_patient_prediction(
        records=eval_records,
        probs=outputs["probabilities"],
        k=int(agg_params["top_k"]),
        threshold=float(agg_params["threshold"]),
        method=str(agg_params["method"]),
        min_suspicious_slices=int(agg_params["min_suspicious_slices"]),
        suspicious_prob_threshold=float(agg_params["suspicious_prob_threshold"]),
        min_suspicious_fraction=float(agg_params["min_suspicious_fraction"]),
        healthy_override_topk_max=float(agg_params["healthy_override_topk_max"]),
        healthy_override_max_suspicious_slices=int(agg_params["healthy_override_max_suspicious_slices"]),
        healthy_override_max_suspicious_fraction=float(agg_params["healthy_override_max_suspicious_fraction"]),
        hard_tumor_topk_min=float(agg_params["hard_tumor_topk_min"]),
        hard_tumor_min_suspicious_slices=int(agg_params["hard_tumor_min_suspicious_slices"]),
    )

    patient_true = {}
    for r in eval_records:
        patient_true[str(r["patient_id"]) ] = int(r["label"])

    patient_y_true = [patient_true[pid] for pid in sorted(patient_true.keys())]
    patient_y_pred = [int(patient_preds.get(pid, 0)) for pid in sorted(patient_true.keys())]
    patient_probs = [[1 - p, p] for p in patient_y_pred]

    patient_metrics = compute_classification_metrics(patient_y_true, patient_y_pred, patient_probs)

    report = {
        "checkpoint": str(ckpt),
        "aggregation_config": str(args.aggregation_config),
        "aggregation_params": agg_params,
        "samples_per_dataset": args.samples_per_dataset,
        "total_samples": len(eval_records),
        "overall_label_counts": dict(Counter(outputs["true_labels"])),
        "overall_prediction_counts": dict(Counter(outputs["predicted_labels"])),
        "overall_metrics": {
            "accuracy": float(overall["accuracy"]),
            "precision": float(overall["precision"]),
            "recall": float(overall["recall"]),
            "f1_score": float(overall["f1_score"]),
            "sensitivity": float(overall["sensitivity"]),
            "specificity": float(overall["specificity"]),
            "confusion_matrix": overall["confusion_matrix"].tolist(),
            "roc_auc": float(overall["roc_auc"]),
        },
        "patient_level_metrics": {
            "num_patients": len(patient_y_true),
            "accuracy": float(patient_metrics["accuracy"]),
            "precision": float(patient_metrics["precision"]),
            "recall": float(patient_metrics["recall"]),
            "f1_score": float(patient_metrics["f1_score"]),
            "sensitivity": float(patient_metrics["sensitivity"]),
            "specificity": float(patient_metrics["specificity"]),
            "confusion_matrix": patient_metrics["confusion_matrix"].tolist(),
            "roc_auc": float(patient_metrics["roc_auc"]),
        },
        "per_dataset": per_group,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Overall confusion matrix:", report["overall_metrics"]["confusion_matrix"])
    print("Overall accuracy:", report["overall_metrics"]["accuracy"])
    print("Overall specificity:", report["overall_metrics"]["specificity"])
    print("Report saved:", out_path)


if __name__ == "__main__":
    main()

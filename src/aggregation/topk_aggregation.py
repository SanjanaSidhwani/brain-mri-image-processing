from collections import defaultdict
import numpy as np


def aggregate_patient_tumor_score(
    tumor_probs,
    top_k=20,
    method="median"
):
    if len(tumor_probs) == 0:
        raise ValueError("tumor_probs cannot be empty")

    probs = np.asarray(tumor_probs, dtype=np.float32)
    k = max(1, min(int(top_k), len(probs)))
    topk = np.sort(probs)[-k:]

    if method == "mean":
        return float(np.mean(topk))
    if method == "median":
        return float(np.median(topk))

    raise ValueError(f"Unsupported aggregation method: {method}")


def robust_patient_prediction_from_tumor_probs(
    tumor_probs,
    threshold=0.70,
    top_k=20,
    method="median",
    min_suspicious_slices=8,
    suspicious_prob_threshold=0.90,
    min_suspicious_fraction=0.30,
    healthy_override_topk_max=0.20,
    healthy_override_max_suspicious_slices=3,
    healthy_override_max_suspicious_fraction=0.05,
    hard_tumor_topk_min=0.95,
    hard_tumor_min_suspicious_slices=3,
):
    topk_score = aggregate_patient_tumor_score(
        tumor_probs=tumor_probs,
        top_k=top_k,
        method=method,
    )

    probs = np.asarray(tumor_probs, dtype=np.float32)
    suspicious_count = int(np.sum(probs >= float(suspicious_prob_threshold)))
    suspicious_fraction = float(suspicious_count / max(1, len(probs)))
    if min_suspicious_fraction <= 0:
        fraction_strength = 1.0
    else:
        fraction_strength = min(1.0, suspicious_fraction / float(min_suspicious_fraction))

    risk_score = float(0.5 * topk_score + 0.5 * fraction_strength)

    base_tumor_rule = (
        (topk_score >= float(threshold))
        and (suspicious_count >= int(min_suspicious_slices))
        and (suspicious_fraction >= float(min_suspicious_fraction))
    )

    # Safety valve: if the tumor signal is extremely strong, never override to healthy.
    hard_tumor_rule = (
        (topk_score >= float(hard_tumor_topk_min))
        and (suspicious_count >= int(hard_tumor_min_suspicious_slices))
    )

    # Conservative healthy override for very weak and sparse tumor evidence.
    healthy_override_rule = (
        (topk_score <= float(healthy_override_topk_max))
        and (suspicious_count <= int(healthy_override_max_suspicious_slices))
        and (suspicious_fraction <= float(healthy_override_max_suspicious_fraction))
    )

    if hard_tumor_rule:
        prediction = 1
    elif healthy_override_rule:
        prediction = 0
    else:
        prediction = 1 if base_tumor_rule else 0

    confidence = risk_score if prediction == 1 else max(1.0 - suspicious_fraction, 1.0 - topk_score)

    return {
        "prediction": prediction,
        "score": float(topk_score),
        "risk_score": risk_score,
        "confidence": float(confidence),
        "suspicious_slices": suspicious_count,
        "suspicious_fraction": suspicious_fraction,
        "base_tumor_rule": bool(base_tumor_rule),
        "hard_tumor_rule": bool(hard_tumor_rule),
        "healthy_override_rule": bool(healthy_override_rule),
        "threshold": float(threshold),
        "top_k": int(top_k),
        "method": method,
    }


def topk_patient_prediction(
    records,
    probs,
    k=20,
    threshold=0.70,
    method="median",
    min_suspicious_slices=8,
    suspicious_prob_threshold=0.90,
    min_suspicious_fraction=0.30,
    healthy_override_topk_max=0.20,
    healthy_override_max_suspicious_slices=3,
    healthy_override_max_suspicious_fraction=0.05,
    hard_tumor_topk_min=0.95,
    hard_tumor_min_suspicious_slices=3,
):
    
    patient_dict = defaultdict(list)

    for record, prob in zip(records, probs):
        patient_id = record["patient_id"]
        patient_dict[patient_id].append(prob)

    patient_predictions = {}

    for patient_id, patient_probs in patient_dict.items():

        patient_probs = np.array(patient_probs)

        class1_probs = patient_probs[:, 1]

        decision = robust_patient_prediction_from_tumor_probs(
            tumor_probs=class1_probs,
            threshold=threshold,
            top_k=max(1, int(k)),
            method=method,
            min_suspicious_slices=min_suspicious_slices,
            suspicious_prob_threshold=suspicious_prob_threshold,
            min_suspicious_fraction=min_suspicious_fraction,
            healthy_override_topk_max=healthy_override_topk_max,
            healthy_override_max_suspicious_slices=healthy_override_max_suspicious_slices,
            healthy_override_max_suspicious_fraction=healthy_override_max_suspicious_fraction,
            hard_tumor_topk_min=hard_tumor_topk_min,
            hard_tumor_min_suspicious_slices=hard_tumor_min_suspicious_slices,
        )
        prediction = decision["prediction"]

        patient_predictions[patient_id] = prediction

    return patient_predictions

def get_patient_labels(records):

    patient_labels = {}

    for record in records:
        patient_id = record["patient_id"]
        label = record["label"]

        patient_labels[patient_id] = label  

    return patient_labels
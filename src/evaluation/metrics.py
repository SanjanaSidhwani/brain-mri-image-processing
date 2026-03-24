from typing import Dict, List

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)


def compute_classification_metrics(
    true_labels: List[int],
    predicted_labels: List[int],
    probabilities: List[List[float]] = None
) -> Dict:

    accuracy = accuracy_score(true_labels, predicted_labels)

    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)

    cm = confusion_matrix(true_labels, predicted_labels)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
    }

    
    if probabilities is not None:
        class1_probs = [p[1] for p in probabilities]
        roc_auc = roc_auc_score(true_labels, class1_probs)
        metrics["roc_auc"] = roc_auc

    return metrics
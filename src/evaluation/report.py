import json
from pathlib import Path
from typing import Dict, Optional


def print_report(metrics: Dict) -> None:

    print("\n" + "=" * 50)
    print("===== Brain MRI Classification Report =====")
    print("=" * 50 + "\n")

    accuracy = metrics.get("accuracy", 0.0)
    precision = metrics.get("precision", 0.0)
    recall = metrics.get("recall", 0.0)
    f1 = metrics.get("f1_score", 0.0)
    cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}\n")

    if len(cm) != 2 or len(cm[0]) != 2:
        raise ValueError("Confusion matrix must be 2x2 for binary classification")

    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]

    print("Confusion Matrix:")
    print("-" * 35)
    print("            Predicted")
    print("              0         1")
    print(f"Actual  0  {tn:>7}    {fp:>7}")
    print(f"        1  {fn:>7}    {tp:>7}")
    print("-" * 35 + "\n")


def save_report(metrics: Dict, filepath: str) -> None:

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    metrics_to_save = metrics.copy()

    cm = metrics_to_save.get("confusion_matrix", [[0, 0], [0, 0]])

    if hasattr(cm, "tolist"):
        metrics_to_save["confusion_matrix"] = cm.tolist()
    else:
        metrics_to_save["confusion_matrix"] = [list(row) for row in cm]

    with open(filepath, "w") as f:
        json.dump(metrics_to_save, f, indent=2)


def generate_report(metrics: Dict, save_path: Optional[str] = None) -> None:

    print_report(metrics)

    if save_path:
        save_report(metrics, save_path)
        print(f"Report saved to: {save_path}\n")
import numpy as np

preds = np.load("outputs/preds.npy")
labels = np.load("outputs/labels.npy")

errors = preds != labels

total_samples = len(labels)
total_errors = errors.sum()
error_rate = total_errors / total_samples

print(f"Total Samples: {total_samples}")
print(f"Total Errors: {total_errors}")
print(f"Error Rate: {error_rate:.4f}")
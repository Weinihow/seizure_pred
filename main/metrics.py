'''
YCWang
Sept.18, 2025
This code provides evaluation metrics
metric list:
    accuracy
    recall (sensitivity)
    precision
    specificity
    false positive rate
    AUC
    F1-score (precision & recall)
'''
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
import os
import datetime

def basic_metric(y_true, y_pred, y_score=None, metrics=['all']):
    def normalize(name):
        return name.lower().replace("-", "_")
    metrics = [normalize(m) for m in metrics]   # make all metric selection same form

    if metrics == ['all']:
        metrics = ["accuracy", "recall", "precision", "specificity", "fpr", "auc", "f1_score"]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel().tolist()

    results = {}
    if "accuracy" in metrics:
        results["accuracy"] = (tp + tn) / (tn + tp + fp + fn)
    if "recall" in metrics:
        results["recall"] = tp / (tp + fn) if (tp + fn) else 0
    if "precision" in metrics:
        results["precision"] = tp / (tp + fp) if (tp + fp) else 0
    if "specificity" in metrics:
        results["specificity"] = tn / (tn + fp) if (tn + fp) else 0
    if "auc" in metrics:
        if y_score is None:
            raise ValueError("y_score is required to compute AUC")
        results["AUC"] = roc_auc_score(y_true, y_score)
    if "f1_score" in metrics:
        results["F1_score"] = f1_score(y_true, y_pred)
    if "fpr" in metrics:
        results["fpr"] = fp / (fp + tn)

    max_len = max(len(k) for k in results.keys())
    for k, v in results.items():
        print(f"{k.ljust(max_len)}: {v:.4f}")

    return results

def log_results_to_md(filename, model_name, model_type, threshold, results, training_duration=None):
    """
    Log training results to a markdown file.
    Table columns: Time, Model Name, Type, Threshold, F1, Accuracy, Recall, FPR, Duration
    """
    header = "| Time | Model Name | Type | Threshold | F1 | Accuracy | Recall | FPR | Duration |\n|---|---|---|---|---|---|---|---|---|\n"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Check if file exists to check header or write new
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write(header)
            
    # Prepare row
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    f1 = results.get('F1_score', 0)
    acc = results.get('accuracy', 0)
    rec = results.get('recall', 0)
    fpr = results.get('fpr', 0)
    
    # Format duration
    duration_str = "-"
    if training_duration is not None:
        m, s = divmod(training_duration, 60)
        h, m = divmod(m, 60)
        duration_str = f"{int(h)}h {int(m)}m {int(s)}s"
    
    row = f"| {now} | {model_name} | {model_type} | {threshold:.4f} | {f1:.4f} | {acc:.4f} | {rec:.4f} | {fpr:.4f} | {duration_str} |\n"
    
    with open(filename, 'a') as f:
        f.write(row)
    print(f"Logged results to {filename}")

if __name__ == '__main__':
    y_true = [0, 0, 1, 1, 0, 1, 0, 1]
    y_pred = [0, 0, 1, 0, 1, 1, 1, 1]
    y_score = [0.1, 0.3, 0.9, 0.4, 0.2, 0.8, 0.7, 0.95]
    results = basic_metric(y_true, y_pred, y_score, metrics=['all'])
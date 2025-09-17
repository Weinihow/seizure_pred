'''
YCWang
Sept.18, 2025
This code provides evaluation metrics
metric list:
    accuracy
    recall (sensitivity)
    precision
    specificity
    AUC
    F1-score (precision & recall)
'''
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score

def basic_metric(y_true, y_pred, y_score=None, metrics=['all']):
    def normalize(name):
        return name.lower().replace("-", "_")
    metrics = [normalize(m) for m in metrics]   # make all metric selection same form

    if metrics == ['all']:
        metrics = ["accuracy", "recall", "precision", "specificity", "auc", "f1_score"]

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

    max_len = max(len(k) for k in results.keys())
    for k, v in results.items():
        print(f"{k.ljust(max_len)}: {v}")

    return results
    
if __name__ == '__main__':
    y_true = [0, 0, 1, 1, 0, 1, 0, 1]
    y_pred = [0, 0, 1, 0, 1, 1, 1, 1]
    y_score = [0.1, 0.3, 0.9, 0.4, 0.2, 0.8, 0.7, 0.95]
    results = basic_metric(y_true, y_pred, y_score, metrics=['ALL'])
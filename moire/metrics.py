from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class EvalResult:
    accuracy: float
    f1: float
    precision: float
    recall: float
    auc: float
    tn: int
    fp: int
    fn: int
    tp: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "accuracy": float(self.accuracy),
            "f1": float(self.f1),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "auc": float(self.auc),
            "tn": int(self.tn),
            "fp": int(self.fp),
            "fn": int(self.fn),
            "tp": int(self.tp),
        }


def compute_binary_metrics(y_true, y_prob, threshold: float = 0.5) -> EvalResult:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= float(threshold)).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return EvalResult(acc, f1, prec, rec, auc, int(tn), int(fp), int(fn), int(tp))

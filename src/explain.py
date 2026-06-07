"""Wyjasnienia SHAP (TreeSHAP) dla modeli drzewiastych."""

import numpy as np
import pandas as pd
import shap

SEED = 42


def _stack(vals):
    """Ujednolica wynik shap_values do tablicy (n_samples, n_features, n_classes)."""
    if isinstance(vals, list):
        return np.stack([np.asarray(v) for v in vals], axis=-1)
    vals = np.asarray(vals)
    return vals if vals.ndim == 3 else vals[:, :, None]


def shap_importance(model, X, features, sample=2000):
    Xs = X.sample(min(sample, len(X)), random_state=SEED)
    arr = _stack(shap.TreeExplainer(model).shap_values(Xs))  # (n, f, C)
    global_imp = np.abs(arr).mean(axis=(0, 2))
    ranking = (pd.DataFrame({"feature": features, "importance": global_imp})
               .sort_values("importance", ascending=False).reset_index(drop=True))
    per_class = np.abs(arr).mean(axis=0)  # (f, C)
    return ranking, per_class


def per_class_top(per_class, features, classes, k=5):
    out = {}
    for j, cls in enumerate(classes):
        order = np.argsort(per_class[:, j])[::-1][:k]
        out[cls] = [features[i] for i in order]
    return out

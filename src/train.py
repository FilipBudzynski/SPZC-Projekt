"""Trening Random Forest i XGBoost oraz metryki dla klasyfikacji wieloklasowej."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, classification_report)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

SEED = 42


def build_models():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.3,
            tree_method="hist", n_jobs=-1, random_state=SEED),
    }


def fit(model, X_train, y_train):
    if isinstance(model, XGBClassifier):
        model.fit(X_train, y_train,
                  sample_weight=compute_sample_weight("balanced", y_train))
    else:
        model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, classes):
    pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision_macro": precision_score(y_test, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0),
    }
    report = classification_report(y_test, pred, target_names=classes,
                                   zero_division=0, output_dict=True)
    cm = confusion_matrix(y_test, pred)
    return metrics, report, cm

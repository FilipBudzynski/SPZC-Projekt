"""Eksperyment: RF i XGBoost + SHAP na UNSW-NB15 i CIC-IoT2023 (wieloklasowo)."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.metrics import ConfusionMatrixDisplay

from src.data import LOADERS
from src.explain import shap_importance, per_class_top
from src.train import build_models, fit, evaluate

RESULTS = "results"
TOP_K = 15


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def plot_importance(ranking, title, path):
    top = ranking.head(TOP_K).iloc[::-1]
    plt.figure(figsize=(7, 5))
    plt.barh(top["feature"], top["importance"], color="#3b6ea5")
    plt.xlabel("Srednia |wartosc SHAP|")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_cm(cm, classes, title, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(
        ax=ax, cmap="Blues", xticks_rotation=90, colorbar=False, values_format="d")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run():
    os.makedirs(RESULTS, exist_ok=True)
    summary, rankings, per_class_out = {}, {}, {}

    for ds_name, loader in LOADERS.items():
        print(f"\n=== {ds_name} ===")
        ds = loader()
        print(f"train={ds.X_train.shape} test={ds.X_test.shape} klasy={len(ds.classes)}")
        summary[ds_name] = {}

        for m_name, model in build_models().items():
            print(f"  -> {m_name}")
            model = fit(model, ds.X_train, ds.y_train)
            metrics, report, cm = evaluate(model, ds.X_test, ds.y_test, ds.classes)

            ranking, per_class = shap_importance(model, ds.X_test, ds.features)
            top_feats = ranking["feature"].head(TOP_K).tolist()

            top_model = fit(clone(model), ds.X_train[top_feats], ds.y_train)
            top_metrics, _, _ = evaluate(top_model, ds.X_test[top_feats], ds.y_test, ds.classes)

            summary[ds_name][m_name] = {"all_features": metrics, "top15_features": top_metrics}
            rankings[(ds_name, m_name)] = top_feats
            per_class_out[f"{ds_name}/{m_name}"] = per_class_top(per_class, ds.features, ds.classes)

            tag = f"{ds_name}_{m_name}"
            ranking.to_csv(f"{RESULTS}/{tag}_shap.csv", index=False)
            json.dump(report, open(f"{RESULTS}/{tag}_report.json", "w"), indent=2)
            plot_importance(ranking, f"SHAP top {TOP_K} - {m_name} ({ds_name})",
                            f"{RESULTS}/{tag}_shap.png")
            plot_cm(cm, ds.classes, f"Macierz pomylek - {m_name} ({ds_name})",
                    f"{RESULTS}/{tag}_cm.png")

    comparison = {}
    for ds_name in LOADERS:
        rf, xgb = rankings[(ds_name, "RandomForest")], rankings[(ds_name, "XGBoost")]
        comparison[ds_name] = {
            "rf_top15": rf, "xgb_top15": xgb,
            "jaccard_top15": round(jaccard(rf, xgb), 3),
            "common": sorted(set(rf) & set(xgb)),
        }

    json.dump(summary, open(f"{RESULTS}/metrics.json", "w"), indent=2)
    json.dump(comparison, open(f"{RESULTS}/comparison.json", "w"), indent=2)
    json.dump(per_class_out, open(f"{RESULTS}/per_class_top.json", "w"), indent=2)

    print("\n=== PODSUMOWANIE ===")
    for ds_name in summary:
        for m_name in summary[ds_name]:
            a = summary[ds_name][m_name]["all_features"]
            t = summary[ds_name][m_name]["top15_features"]
            print(f"{ds_name:12} {m_name:13} acc={a['accuracy']:.4f} f1m={a['f1_macro']:.4f} "
                  f"| top15 acc={t['accuracy']:.4f} f1m={t['f1_macro']:.4f}")
        print(f"  RF vs XGB Jaccard(top15) = {comparison[ds_name]['jaccard_top15']}")


if __name__ == "__main__":
    run()

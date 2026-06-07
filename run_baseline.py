"""Baseline reprodukcyjny: ten sam potok na NSL-KDD (zbiorze z pracy bazowej)."""

import json
import os

from sklearn.base import clone

from run import RESULTS, TOP_K, jaccard, plot_importance, plot_cm
from src.data import BASELINE_LOADERS
from src.explain import shap_importance, per_class_top
from src.train import build_models, fit, evaluate


def run():
    os.makedirs(RESULTS, exist_ok=True)
    metrics = json.load(open(f"{RESULTS}/metrics.json"))
    comparison = json.load(open(f"{RESULTS}/comparison.json"))
    per_class_out = json.load(open(f"{RESULTS}/per_class_top.json"))

    for ds_name, loader in BASELINE_LOADERS.items():
        print(f"\n=== {ds_name} (baseline) ===")
        ds = loader()
        print(f"train={ds.X_train.shape} test={ds.X_test.shape} klasy={len(ds.classes)}")
        metrics[ds_name] = {}
        rankings = {}

        for m_name, model in build_models().items():
            print(f"  -> {m_name}")
            model = fit(model, ds.X_train, ds.y_train)
            m, report, cm = evaluate(model, ds.X_test, ds.y_test, ds.classes)

            ranking, per_class = shap_importance(model, ds.X_test, ds.features)
            top_feats = ranking["feature"].head(TOP_K).tolist()

            top_model = fit(clone(model), ds.X_train[top_feats], ds.y_train)
            top_m, _, _ = evaluate(top_model, ds.X_test[top_feats], ds.y_test, ds.classes)

            metrics[ds_name][m_name] = {"all_features": m, "top15_features": top_m}
            rankings[m_name] = top_feats
            per_class_out[f"{ds_name}/{m_name}"] = per_class_top(per_class, ds.features, ds.classes)

            tag = f"{ds_name}_{m_name}"
            ranking.to_csv(f"{RESULTS}/{tag}_shap.csv", index=False)
            json.dump(report, open(f"{RESULTS}/{tag}_report.json", "w"), indent=2)
            plot_importance(ranking, f"SHAP top {TOP_K} - {m_name} ({ds_name})",
                            f"{RESULTS}/{tag}_shap.png")
            plot_cm(cm, ds.classes, f"Macierz pomylek - {m_name} ({ds_name})",
                    f"{RESULTS}/{tag}_cm.png")

        rf, xgb = rankings["RandomForest"], rankings["XGBoost"]
        comparison[ds_name] = {
            "rf_top15": rf, "xgb_top15": xgb,
            "jaccard_top15": round(jaccard(rf, xgb), 3),
            "common": sorted(set(rf) & set(xgb)),
        }

    json.dump(metrics, open(f"{RESULTS}/metrics.json", "w"), indent=2)
    json.dump(comparison, open(f"{RESULTS}/comparison.json", "w"), indent=2)
    json.dump(per_class_out, open(f"{RESULTS}/per_class_top.json", "w"), indent=2)

    print("\n=== BASELINE ===")
    for ds_name in BASELINE_LOADERS:
        for m_name in metrics[ds_name]:
            a = metrics[ds_name][m_name]["all_features"]
            print(f"{ds_name:10} {m_name:13} acc={a['accuracy']:.4f} "
                  f"f1m={a['f1_macro']:.4f} f1w={a['f1_weighted']:.4f}")


if __name__ == "__main__":
    run()

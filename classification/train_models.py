"""
TransitMind - Classification comparison

Label  : activity_class = Low / Medium / High, terciles of trips_per_hour
         (scheduled service frequency of a route in a time period).
         These are OPERATIONAL ACTIVITY classes, not passenger-demand classes.
Inputs : route structure + time period (no trip-count columns -> no leakage).
Split  : 75/25 with whole routes kept together.
Output : data/processed/classification_results.csv,
         data/processed/classification_confusion.csv, models/classifier.joblib
"""
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing.feature_engineering import (  # noqa: E402
    CLASS_LABELS, CLASSIFICATION_TARGET, DATA_PROCESSED, FEATURES, MODELS_DIR,
    RANDOM_STATE, group_cv, group_split, load_ml_dataset, make_pipeline)

RESULTS_FILE = DATA_PROCESSED / "classification_results.csv"
CONFUSION_FILE = DATA_PROCESSED / "classification_confusion.csv"
MODEL_FILE = MODELS_DIR / "classifier.joblib"


def build_models():
    svc = lambda **kw: SVC(random_state=RANDOM_STATE, **kw)   # noqa: E731
    return {
        "KNN (k=7)": make_pipeline(KNeighborsClassifier(n_neighbors=7)),
        # C large -> narrow margin / almost no violations allowed ("hard-margin-like")
        "Linear SVM (C=100)": make_pipeline(svc(kernel="linear", C=100.0)),
        # C small -> wide margin, violations tolerated ("soft margin")
        "Soft-margin SVM (linear, C=0.1)": make_pipeline(svc(kernel="linear", C=0.1)),
        "RBF SVM": make_pipeline(svc(kernel="rbf", C=10.0, gamma="scale")),
        "Polynomial SVM (deg 3)": make_pipeline(svc(kernel="poly", degree=3, C=1.0, gamma="scale")),
        "Sigmoid SVM": make_pipeline(svc(kernel="sigmoid", C=1.0, gamma="scale")),
    }


def main():
    warnings.filterwarnings("ignore")
    print("=" * 60)
    print("TransitMind - Classification (operational activity class)")
    print("=" * 60)
    df = load_ml_dataset()
    counts = df[CLASSIFICATION_TARGET].value_counts()
    print("Class counts:", counts.to_dict())
    if counts.size < 2 or counts.min() < 10:
        sys.exit("ERROR: need at least 2 classes with >=10 samples each.")
    labels = [c for c in CLASS_LABELS if c in counts.index]

    train, test = group_split(df)
    if train[CLASSIFICATION_TARGET].nunique() < 2:
        sys.exit("ERROR: training split contains a single class.")
    X_tr, y_tr = train[FEATURES], train[CLASSIFICATION_TARGET]
    X_te, y_te = test[FEATURES], test[CLASSIFICATION_TARGET]
    cv = group_cv(train)

    rows, matrices, fitted = [], [], {}
    for name, model in build_models().items():
        t0 = time.time()
        try:
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
            cv_acc = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="accuracy").mean()
        except Exception as exc:
            print(f"  ! {name} failed: {exc}")
            continue
        kw = dict(average="weighted", zero_division=0)
        rows.append({"model": name, "Accuracy": accuracy_score(y_te, pred),
                     "Precision": precision_score(y_te, pred, **kw),
                     "Recall": recall_score(y_te, pred, **kw),
                     "F1": f1_score(y_te, pred, **kw), "CV_Accuracy_mean": cv_acc,
                     "train_seconds": round(time.time() - t0, 2)})
        cm = confusion_matrix(y_te, pred, labels=labels)
        for i, actual in enumerate(labels):
            for j, predicted in enumerate(labels):
                matrices.append({"model": name, "actual": actual,
                                 "predicted": predicted, "count": int(cm[i, j])})
        fitted[name] = model
        print(f"  {name:<34} acc={rows[-1]['Accuracy']:.3f}  F1={rows[-1]['F1']:.3f}")
    if not rows:
        sys.exit("ERROR: no classifier could be trained.")

    res = pd.DataFrame(rows).sort_values("F1", ascending=False).round(4)
    best = res.iloc[0]["model"]
    MODELS_DIR.mkdir(exist_ok=True)
    res.to_csv(RESULTS_FILE, index=False)
    pd.DataFrame(matrices).to_csv(CONFUSION_FILE, index=False)
    joblib.dump({"models": fitted, "best": best, "labels": labels,
                 "features": FEATURES}, MODEL_FILE)
    print(f"\nBest model by test F1: {best}")
    print(f"Saved {RESULTS_FILE}\nSaved {CONFUSION_FILE}\nSaved {MODEL_FILE}")


if __name__ == "__main__":
    main()

"""
TransitMind - Does combining models beat a single model?

Same data, same route-grouped train/test split and same features as the
regression and classification modules.

Regression    (scheduled_trips_in_period): single models vs Voting, Averaging,
              Weighted averaging, Bagging, Random Forest, AdaBoost,
              Gradient Boosting, Stacking.
Classification (activity_class): single models vs Hard voting, Soft voting
              (= probability averaging), Weighted soft voting, Bagging,
              Random Forest, AdaBoost, Gradient Boosting, Stacking.

Voting/stacking models hold complete Pipelines as base learners (each does its own
encoding/scaling), so they receive the raw DataFrame.

Weights for weighted averaging come from cross-validation on the TRAINING set
only; the test set is never used to choose anything.
Output: data/processed/ensemble_results.csv
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier,
                              BaggingRegressor, GradientBoostingClassifier,
                              GradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor, StackingClassifier, StackingRegressor,
                              VotingClassifier, VotingRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing.feature_engineering import (  # noqa: E402
    CLASSIFICATION_TARGET, DATA_PROCESSED, FEATURES, RANDOM_STATE, REGRESSION_TARGET,
    group_cv, group_split, load_ml_dataset, make_pipeline)

RESULTS_FILE = DATA_PROCESSED / "ensemble_results.csv"
RS = RANDOM_STATE


def reg_bases():
    return [
        ("ridge", make_pipeline(Ridge(alpha=10.0))),
        ("tree", make_pipeline(DecisionTreeRegressor(max_depth=6, min_samples_leaf=5,
                                                     random_state=RS), scale=False)),
        ("svr", make_pipeline(TransformedTargetRegressor(
            regressor=SVR(C=10.0), transformer=StandardScaler()))),
    ]


def cls_bases():
    return [
        ("knn", make_pipeline(KNeighborsClassifier(n_neighbors=7))),
        ("tree", make_pipeline(DecisionTreeClassifier(max_depth=6, min_samples_leaf=5,
                                                      random_state=RS), scale=False)),
        ("rbf", make_pipeline(SVC(kernel="rbf", C=10.0, probability=True, random_state=RS))),
    ]


def run_regression(train, test, cv):
    X_tr, y_tr, X_te, y_te = train[FEATURES], train[REGRESSION_TARGET], test[FEATURES], test[REGRESSION_TARGET]
    bases = reg_bases()
    models = {f"[single] {n}": m for n, m in bases}
    models["Random Forest"] = make_pipeline(RandomForestRegressor(
        n_estimators=100, max_depth=8, min_samples_leaf=3, random_state=RS, n_jobs=1), scale=False)

    models["Voting (mean of 3 models)"] = VotingRegressor(reg_bases())
    models["Bagging (50 trees)"] = make_pipeline(BaggingRegressor(
        DecisionTreeRegressor(max_depth=8, min_samples_leaf=3, random_state=RS),
        n_estimators=50, random_state=RS, n_jobs=1), scale=False)
    models["AdaBoost"] = make_pipeline(AdaBoostRegressor(
        DecisionTreeRegressor(max_depth=4, random_state=RS), n_estimators=50,
        random_state=RS), scale=False)
    models["Gradient Boosting"] = make_pipeline(GradientBoostingRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8,
        random_state=RS), scale=False)
    models["Stacking (Ridge meta-model)"] = StackingRegressor(
        reg_bases(), final_estimator=Ridge(alpha=1.0), cv=cv)

    rows, preds = [], {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        preds[name] = model.predict(X_te)

    # Weighted averaging - weights from out-of-fold R2 on training data
    oof, w = {}, []
    for n, m in bases:
        oof[n] = cross_val_predict(m, X_tr, y_tr, cv=cv)
        w.append(max(r2_score(y_tr, oof[n]), 0.0))
    w = np.array(w) if np.sum(w) > 0 else np.ones(len(bases))
    w = w / w.sum()
    preds["Weighted averaging (CV-R2 weights)"] = sum(
        wi * preds[f"[single] {n}"] for wi, (n, _) in zip(w, bases))
    preds["Averaging (equal weights)"] = np.mean(
        [preds[f"[single] {n}"] for n, _ in bases], axis=0)
    print(f"  weights (ridge, tree, svr): {np.round(w, 3)}")

    for name, p in preds.items():
        mse = mean_squared_error(y_te, p)
        rows.append({"task": "regression", "model": name,
                     "type": "single" if name.startswith("[single]") else "ensemble",
                     "MAE": mean_absolute_error(y_te, p), "RMSE": np.sqrt(mse),
                     "R2": r2_score(y_te, p), "primary_metric": "R2",
                     "primary_score": r2_score(y_te, p)})
    return rows


def run_classification(train, test, cv):
    X_tr, y_tr, X_te, y_te = train[FEATURES], train[CLASSIFICATION_TARGET], test[FEATURES], test[CLASSIFICATION_TARGET]
    bases = cls_bases()
    models = {f"[single] {n}": m for n, m in bases}
    models["Random Forest"] = make_pipeline(RandomForestClassifier(
        n_estimators=100, max_depth=8, min_samples_leaf=3, random_state=RS, n_jobs=1), scale=False)

    models["Hard voting"] = VotingClassifier(cls_bases(), voting="hard")
    models["Soft voting (probability averaging)"] = VotingClassifier(cls_bases(), voting="soft")

    # weights from out-of-fold accuracy on the training set
    w = np.array([max(accuracy_score(y_tr, cross_val_predict(m, X_tr, y_tr, cv=cv)) - 1 / 3, 0.0)
                  for _, m in bases])
    w = w / w.sum() if w.sum() > 0 else np.ones(len(bases)) / len(bases)
    print(f"  weights (knn, tree, rbf): {np.round(w, 3)}")
    models["Weighted soft voting (CV weights)"] = VotingClassifier(
        cls_bases(), voting="soft", weights=list(w))
    models["Bagging (50 trees)"] = make_pipeline(BaggingClassifier(
        DecisionTreeClassifier(max_depth=8, min_samples_leaf=3, random_state=RS),
        n_estimators=50, random_state=RS, n_jobs=1), scale=False)
    models["AdaBoost"] = make_pipeline(AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=3, random_state=RS), n_estimators=50,
        random_state=RS), scale=False)
    models["Gradient Boosting"] = make_pipeline(GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8,
        random_state=RS), scale=False)
    models["Stacking (logistic meta-model)"] = StackingClassifier(
        cls_bases(), final_estimator=LogisticRegression(max_iter=1000), cv=cv)

    rows = []
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        p = model.predict(X_te)
        rows.append({"task": "classification", "model": name,
                     "type": "single" if name.startswith("[single]") else "ensemble",
                     "Accuracy": accuracy_score(y_te, p),
                     "F1": f1_score(y_te, p, average="weighted", zero_division=0),
                     "primary_metric": "F1",
                     "primary_score": f1_score(y_te, p, average="weighted", zero_division=0)})
    return rows


def main():
    warnings.filterwarnings("ignore")
    print("=" * 60)
    print("TransitMind - Ensemble vs single models")
    print("=" * 60)
    df = load_ml_dataset()
    train, test = group_split(df)
    cv = group_cv(train)
    print(f"Train {len(train):,} rows / Test {len(test):,} rows (route-grouped)")

    print("\nRegression:")
    rows = run_regression(train, test, cv)
    print("\nClassification:")
    rows += run_classification(train, test, cv)

    res = pd.DataFrame(rows).round(4)
    res = res.sort_values(["task", "primary_score"], ascending=[True, False])
    res["rank_in_task"] = res.groupby("task")["primary_score"].rank(ascending=False, method="min").astype(int)
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTS_FILE, index=False)

    for task in ("regression", "classification"):
        t = res[res.task == task]
        best_single = t[t.type == "single"].iloc[0]
        best_ens = t[t.type == "ensemble"].iloc[0]
        gain = best_ens.primary_score - best_single.primary_score
        print(f"\n{task.upper()}: best single = {best_single.model} ({best_single.primary_metric}="
              f"{best_single.primary_score:.3f}); best ensemble = {best_ens.model} "
              f"({best_ens.primary_score:.3f}); difference = {gain:+.3f}")
        print(t[["model", "type", "primary_metric", "primary_score"]].to_string(index=False))
    print(f"\nSaved {RESULTS_FILE}")


if __name__ == "__main__":
    main()

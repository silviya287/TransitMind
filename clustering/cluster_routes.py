"""
TransitMind - Route clustering (route level, 314 routes)

Algorithms: K-Means, Hierarchical (Ward), DBSCAN, Gaussian Mixture.
K-Medoids is intentionally NOT included: scikit-learn has no K-Medoids and the
project avoids extra dependencies.
Evaluation: silhouette score, elbow (K-Means inertia) and GMM BIC for k = 2..8.
Cluster descriptions are generated from cluster statistics (not hand-written).

Output: data/processed/route_clusters.csv
        data/processed/clustering_metrics.csv
        data/processed/cluster_profiles.csv
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing.feature_engineering import (  # noqa: E402
    DATA_PROCESSED, RANDOM_STATE, check_file, ML_DATASET_FILE)

CLUSTER_FILE = DATA_PROCESSED / "route_clusters.csv"
METRICS_FILE = DATA_PROCESSED / "clustering_metrics.csv"
PROFILE_FILE = DATA_PROCESSED / "cluster_profiles.csv"

# route-level features used for clustering
FEATURES = ["total_trips", "average_stops", "average_trip_duration",
            "unique_stops", "peak_share", "route_complexity"]
LOG_FEATURES = ["total_trips", "unique_stops"]        # right-skewed counts
WORDS = {"total_trips": ("low frequency", "high frequency"),
         "average_stops": ("few stops per trip", "many stops per trip"),
         "average_trip_duration": ("short trips", "long trips"),
         "unique_stops": ("small network", "large network"),
         "peak_share": ("flat daily profile", "peak-heavy"),
         "route_complexity": ("simple (single pattern)", "complex (many variants)")}


def load_route_table():
    check_file(ML_DATASET_FILE, "Run: python preprocessing\\create_ml_dataset.py")
    df = pd.read_csv(ML_DATASET_FILE, dtype={"route_id": str, "route_short_name": str})
    need = ["route_id", "route_short_name", "total_trips", "average_stops",
            "average_trip_duration", "unique_stops", "route_complexity",
            "morning_trip_count", "evening_trip_count"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        sys.exit(f"ERROR: missing columns {miss}")
    r = df.drop_duplicates("route_id")[need].copy()
    r["peak_share"] = (r["morning_trip_count"] + r["evening_trip_count"]) / r["total_trips"].clip(lower=1)
    r = r.dropna(subset=FEATURES)
    if len(r) < 20:
        sys.exit(f"ERROR: only {len(r)} routes - too few to cluster.")
    return r.reset_index(drop=True)


def scaled_matrix(r):
    x = r[FEATURES].copy()
    for c in LOG_FEATURES:
        x[c] = np.log1p(x[c])
    return StandardScaler().fit_transform(x)


def safe_silhouette(X, labels):
    mask = labels != -1                       # DBSCAN noise is excluded
    if len(set(labels[mask])) < 2 or mask.sum() <= len(set(labels[mask])):
        return np.nan
    return float(silhouette_score(X[mask], labels[mask]))


def describe_clusters(r, label_col):
    """Plain-language description derived from cluster means vs. the overall mean."""
    overall_mean, overall_std = r[FEATURES].mean(), r[FEATURES].std().replace(0, 1)
    rows = []
    for lab, g in r[r[label_col] != -1].groupby(label_col):
        z = (g[FEATURES].mean() - overall_mean) / overall_std
        traits = [WORDS[f][1] if z[f] > 0.5 else WORDS[f][0] for f in FEATURES if abs(z[f]) > 0.5]
        text = ", ".join(traits) if traits else "close to the network average on every feature"
        rows.append({"cluster": int(lab), "routes": len(g),
                     **{f"mean_{f}": round(g[f].mean(), 2) for f in FEATURES},
                     "description": text})
    prof = pd.DataFrame(rows)
    # readable names ordered by frequency: A = most trips
    order = prof.sort_values("mean_total_trips", ascending=False)["cluster"].tolist()
    names = {c: f"Cluster {chr(65 + i)}" for i, c in enumerate(order)}
    prof["cluster_name"] = prof["cluster"].map(names)
    return prof.sort_values("cluster_name"), names


def main():
    warnings.filterwarnings("ignore")
    print("=" * 60)
    print("TransitMind - Route clustering")
    print("=" * 60)
    r = load_route_table()
    X = scaled_matrix(r)
    print(f"Routes: {len(r)} | features: {FEATURES}")

    metrics = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        gm = GaussianMixture(n_components=k, random_state=RANDOM_STATE).fit(X)
        metrics.append({"algorithm": "K-Means (elbow sweep)", "k": k,
                        "silhouette": safe_silhouette(X, km.labels_),
                        "inertia": km.inertia_, "bic": np.nan})
        metrics.append({"algorithm": "GMM (BIC sweep)", "k": k,
                        "silhouette": safe_silhouette(X, gm.predict(X)),
                        "inertia": np.nan, "bic": gm.bic(X)})
    m = pd.DataFrame(metrics)
    km_sweep = m[m.algorithm.str.startswith("K-Means")].dropna(subset=["silhouette"])
    k_best = int(km_sweep.loc[km_sweep.silhouette.idxmax(), "k"])
    # Prefer a k that gives readable segments: take best silhouette in 3..5 if close
    k_view = k_best if 3 <= k_best <= 5 else int(
        km_sweep[km_sweep.k.between(3, 5)].sort_values("silhouette").iloc[-1]["k"])
    print(f"Best silhouette k={k_best}; using k={k_view} for readable segments")

    out = r[["route_id", "route_short_name"] + FEATURES].copy()
    results = []

    km = KMeans(n_clusters=k_view, n_init=10, random_state=RANDOM_STATE).fit(X)
    out["kmeans"] = km.labels_
    results.append(("K-Means", k_view, km.labels_))

    hc = AgglomerativeClustering(n_clusters=k_view, linkage="ward").fit(X)
    out["hierarchical"] = hc.labels_
    results.append(("Hierarchical (Ward)", k_view, hc.labels_))

    gm = GaussianMixture(n_components=k_view, random_state=RANDOM_STATE).fit(X)
    out["gmm"] = gm.predict(X)
    results.append(("Gaussian Mixture", k_view, out["gmm"].values))

    # DBSCAN: eps chosen from the 5-NN distance distribution by best silhouette
    min_samples = 5
    dist = NearestNeighbors(n_neighbors=min_samples).fit(X).kneighbors(X)[0][:, -1]
    best = (None, -2, None)
    for q in (50, 60, 70, 80, 90, 95):
        eps = float(np.percentile(dist, q))
        lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
        s = safe_silhouette(X, lab)
        if not np.isnan(s) and s > best[1] and (lab == -1).mean() < 0.3:
            best = (eps, s, lab)
    if best[0] is None:
        print("DBSCAN: no eps gave >=2 clusters with <30% noise; labelling all as noise (-1).")
        db_labels, eps = np.full(len(X), -1), float("nan")
    else:
        eps, _, db_labels = best
    out["dbscan"] = db_labels
    results.append((f"DBSCAN (eps={eps:.2f}, min_samples={min_samples})",
                    len(set(db_labels) - {-1}), db_labels))

    summary = []
    for name, k, lab in results:
        noise = int((np.asarray(lab) == -1).sum())
        summary.append({"algorithm": name, "k": k, "silhouette": safe_silhouette(X, np.asarray(lab)),
                        "noise_points": noise, "inertia": np.nan, "bic": np.nan})
        print(f"  {name:<40} clusters={k}  silhouette={summary[-1]['silhouette']:.3f}  noise={noise}")
    metrics_out = pd.concat([m, pd.DataFrame(summary)], ignore_index=True).round(4)

    # primary clustering (K-Means) -> readable names
    r_lab = r.copy()
    r_lab["kmeans"] = out["kmeans"].values
    prof, names = describe_clusters(r_lab, "kmeans")
    out["cluster"] = out["kmeans"].map(names)
    out["cluster_description"] = out["kmeans"].map(
        dict(zip(prof["cluster"], prof["description"])))

    out.round(4).to_csv(CLUSTER_FILE, index=False)
    metrics_out.to_csv(METRICS_FILE, index=False)
    prof.to_csv(PROFILE_FILE, index=False)
    print("\nCluster profiles (K-Means):")
    print(prof[["cluster_name", "routes", "mean_total_trips", "mean_average_stops",
                "mean_average_trip_duration", "description"]].to_string(index=False))
    print(f"\nSaved {CLUSTER_FILE}\nSaved {METRICS_FILE}\nSaved {PROFILE_FILE}")


if __name__ == "__main__":
    main()

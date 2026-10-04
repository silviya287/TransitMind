# TransitMind 🚌

### AI-Based Adaptive Bus Scheduling & Route Analysis — decision-support prototype

TransitMind analyses the **scheduled** service of PMPML (Pune) bus routes, compares several
machine-learning models, groups similar routes, and uses a small **Q-learning** agent to suggest
a scheduling action to a **human transport operator**.
It does **not** control real buses and does **not** use passenger counts (none are available).

```text
GTFS data -> cleaning -> route analysis -> feature engineering -> regression -> classification
          -> clustering -> ensemble comparison -> Q-learning decision support -> Streamlit dashboard
```

## Data sources

| Source | Content | Use |
|---|---|---|
| PMPML GTFS feed (2026) | routes, trips, stops, stop-times | Everything except revenue |
| `PMPML_OneDay_Data.xlsx` | 337 revenue records, **one day (17 Feb 2019)**, routes 2 and 27 | Historical context only |

Raw data is excluded from Git (`data/raw/*`).

## Pipeline / how to run (Windows CMD)

```cmd
cd C:\Users\<you>\Documents\TransitMind
.venv\Scripts\activate
pip install -r requirements.txt

python preprocessing\build_dataset.py       :: only if pmpml_operational_data.csv is missing
python preprocessing\analyze_routes.py      :: route summary
python preprocessing\process_revenue.py     :: revenue cleaning (337 rows)
python preprocessing\create_ml_dataset.py   :: route x time-period ML dataset
python regression\train_models.py
python classification\train_models.py
python clustering\cluster_routes.py
python ensemble\compare_models.py
python reinforcement\q_learning.py          :: optional args: 301 "Morning Peak"
streamlit run dashboard\app.py
```

## What each module does

* **Preprocessing** – `build_dataset.py` joins GTFS tables (627,199 stop-times);
  `analyze_routes.py` makes the route summary; `process_revenue.py` fixes Sheet2's swapped
  `Time`/`Bus_No` columns and Excel time fractions (e.g. `0.534722` → 12:50);
  `create_ml_dataset.py` collapses stop-times → trips → **route × time period**
  (1,256 rows, 314 routes); `feature_engineering.py` holds shared feature definitions and the
  route-grouped split.
* **Regression** – target `scheduled_trips_in_period` (scheduled service activity, *not* demand).
  Linear, Multiple Linear, Polynomial, Decision Tree, Random Forest, SVR, Ridge, Lasso;
  MAE / MSE / RMSE / R².
* **Classification** – Low / Medium / High activity (terciles of trips per hour).
  KNN, Linear SVM, Soft-margin SVM, RBF / Polynomial / Sigmoid SVM; accuracy, precision, recall,
  F1, confusion matrices.
* **Clustering** – K-Means, Hierarchical (Ward), DBSCAN, Gaussian Mixture on route-level features;
  silhouette + elbow; cluster descriptions are generated from cluster statistics.
  K-Medoids is not included (no lightweight implementation in scikit-learn).
* **Ensemble** – Voting, Averaging, Weighted averaging, Bagging, Random Forest, AdaBoost,
  Gradient Boosting, Stacking vs. single models, with the same split.
* **Q-learning** – tabular, NumPy only; state = (activity, period, condition, fleet level,
  frequency level); actions = keep / add bus / remove bus / increase / decrease frequency.
* **Dashboard** – Overview, Route Analysis, Regression, Classification, Clustering, Ensemble,
  Adaptive Scheduling.

## Results on the current data (test set = routes never seen in training)

Results are reproducible (`random_state=42`) and are **modest** — reported as they are.

* Regression: best Random Forest, R² ≈ 0.30 (linear models ≈ 0.1). Route structure explains only part
  of how many trips are scheduled.
* Classification: best RBF SVM, accuracy ≈ 0.52 (3 classes); gradient boosting reaches F1 ≈ 0.60
  in the ensemble comparison.
* Ensembles: tree-based ensembles beat the single models; simple voting/averaging/stacking did not
  clearly beat the best single model. One split on ~80 test routes → differences are indicative only.
* Clustering: 4 K-Means segments (silhouette ≈ 0.26 — moderate, overlapping groups).
* Q-learning: in its simulator the learned policy scores ≈ +9 per episode vs ≈ −34 (random) and
  ≈ −29 (always keep). This only shows the agent learned the *simulated* objective.

## Real data vs. proxy / simulated values

| Item | Status |
|---|---|
| Trips, stops, durations, start hours | Real **scheduled** GTFS data |
| `scheduled_trips_in_period`, `trips_per_hour`, activity class | Derived from the schedule (not passenger demand) |
| `buses_required_proxy` = trips/hour × duration / 60 | Proxy (vehicles needed to run the timetable) |
| Revenue (2019, routes 2 & 27) | Real but historical, one day; only route 2 exists in the 2026 feed; **not used as a model feature** |
| Operational condition, service-need formula, reward, fleet/frequency levels | **Simulated / proxy** |

## Limitations

1. GTFS contains timetables, not passenger counts, waiting times or occupancy.
2. This is a decision-support prototype; recommendations are advisory only.
3. Revenue data covers one day and two routes, and is from 2019 vs. a 2026 feed — it is not joined by date.
4. Q-learning uses a simulated/proxy environment; its reward is not real passenger waiting time.
5. No real-time feed is integrated (future work).
6. Predictions are estimates, not guaranteed real-world outcomes.

## Future work

Verified real-time/AVL feed, real ridership or ticketing data, validation against observed
waiting times, richer simulator calibrated on real data, time-based model validation, and
operator feedback loops.

"""
TransitMind - Q-Learning decision support (tabular, NumPy only)

ADVISORY ONLY.  The agent is trained on a small SIMULATED environment and its
output is a recommendation for a human transport operator.  It does not
control, and is not connected to, any real bus.

What is real data and what is simulated
---------------------------------------
REAL (from the PMPML GTFS schedule):
    * a route's scheduled activity level in a time period (Low/Medium/High,
      terciles of trips per hour) and its buses_required_proxy
SIMULATED / PROXY (no passenger counts, waiting times or occupancy exist):
    * operational condition (normal / congested / disrupted) and how it changes
    * the "service need" formula below and the reward built on it
    * fleet level and frequency level changes caused by the actions
The reward therefore measures how well supply matches a *proxy* requirement.
It is NOT passenger waiting time or real overcrowding.

State  = (activity 0-2, period 0-3, condition 0-2, bus_level 0-4, freq_level 0-2)
Action = 0 Keep | 1 Add bus | 2 Remove bus | 3 Increase frequency | 4 Decrease frequency
need   = clip(2 + (activity-1) + period_adj + condition_adj, 0, 4)
supply = bus_level + (freq_level - 1)
reward = +2 if supply == need; -2 per level of shortfall (unmet requirement /
         overcrowding proxy); -1 per level of excess (unneeded buses / poor
         utilisation); minus a small cost for changing buses (0.3) or frequency (0.1);
         -0.5 for an impossible move (e.g. adding a bus at the fleet maximum).

Usage:  python reinforcement\\q_learning.py [route_short_name] [time period]
        python reinforcement\\q_learning.py 301 "Morning Peak"
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing.feature_engineering import (  # noqa: E402
    DATA_PROCESSED, ML_DATASET_FILE, MODELS_DIR, RANDOM_STATE, TIME_PERIODS)

Q_FILE = MODELS_DIR / "q_table.npy"
LOG_FILE = DATA_PROCESSED / "qlearning_training_log.csv"

ACTIONS = ["Keep current allocation", "Add bus", "Remove bus",
           "Increase service frequency", "Decrease service frequency"]
ACTIVITY = ["Low", "Medium", "High"]
CONDITIONS = ["Normal", "Congested", "Disrupted"]
BUS_LEVELS = ["Very low", "Below normal", "Normal", "Above normal", "High"]
FREQ_LEVELS = ["Reduced", "Scheduled", "Increased"]
PERIOD_ADJ = {"Morning Peak": 1, "Midday": 0, "Evening Peak": 1, "Off Peak": -1}
COND_ADJ = [0, 1, 1]
# condition Markov chain (rows: current, columns: next)
COND_TRANSITION = np.array([[0.70, 0.20, 0.10],
                            [0.30, 0.55, 0.15],
                            [0.35, 0.25, 0.40]])
N_STATES_SHAPE = (3, 4, 3, 5, 3)
N_ACTIONS = len(ACTIONS)


# ---------------------------------------------------------------- environment
def service_need(activity, period_idx, condition):
    return int(np.clip(2 + (activity - 1) + PERIOD_ADJ[TIME_PERIODS[period_idx]]
                       + COND_ADJ[condition], 0, 4))


def apply_action(bus, freq, action):
    """Return (new_bus, new_freq, action_cost)."""
    if action == 1:
        return (bus + 1, freq, 0.3) if bus < 4 else (bus, freq, 0.5)
    if action == 2:
        return (bus - 1, freq, 0.3) if bus > 0 else (bus, freq, 0.5)
    if action == 3:
        return (bus, freq + 1, 0.1) if freq < 2 else (bus, freq, 0.5)
    if action == 4:
        return (bus, freq - 1, 0.1) if freq > 0 else (bus, freq, 0.5)
    return bus, freq, 0.0


def reward_for(state, action):
    """Deterministic immediate reward and resulting (bus, freq)."""
    activity, period, cond, bus, freq = state
    nb, nf, cost = apply_action(bus, freq, action)
    gap = (nb + nf - 1) - service_need(activity, period, cond)
    base = 2.0 if gap == 0 else (-2.0 * -gap if gap < 0 else -1.0 * gap)
    return base - cost, nb, nf


def step(state, action, rng):
    reward, nb, nf = reward_for(state, action)
    activity, period, cond, _, _ = state
    next_cond = int(rng.choice(3, p=COND_TRANSITION[cond]))
    return (activity, (period + 1) % 4, next_cond, nb, nf), reward


# -------------------------------------------------------------------- learning
def train(episodes=6000, steps=12, alpha=0.1, gamma=0.9, eps_start=1.0,
          eps_end=0.05, seed=RANDOM_STATE, verbose=True):
    rng = np.random.default_rng(seed)
    Q = np.zeros(N_STATES_SHAPE + (N_ACTIONS,))
    decay = (eps_end / eps_start) ** (1 / max(episodes - 1, 1))
    eps, log = eps_start, []
    for ep in range(episodes):
        state = (int(rng.integers(3)), int(rng.integers(4)), int(rng.integers(3)),
                 int(rng.integers(5)), int(rng.integers(3)))
        total = 0.0
        for _ in range(steps):
            a = int(rng.integers(N_ACTIONS)) if rng.random() < eps else int(np.argmax(Q[state]))
            nxt, r = step(state, a, rng)
            Q[state + (a,)] += alpha * (r + gamma * Q[nxt].max() - Q[state + (a,)])
            state, total = nxt, total + r
        log.append({"episode": ep + 1, "total_reward": total, "epsilon": eps})
        eps = max(eps_end, eps * decay)
        if verbose and (ep + 1) % 1000 == 0:
            recent = np.mean([x["total_reward"] for x in log[-200:]])
            print(f"  episode {ep + 1:>5}  mean reward (last 200) = {recent:6.2f}  eps = {eps:.3f}")
    return Q, pd.DataFrame(log)


def evaluate(Q, episodes=500, steps=12, seed=123):
    """Average episode reward: greedy learned policy vs random vs always-keep."""
    out = {}
    for name in ("learned", "random", "keep"):
        rng = np.random.default_rng(seed)
        totals = []
        for _ in range(episodes):
            s = (int(rng.integers(3)), int(rng.integers(4)), int(rng.integers(3)),
                 int(rng.integers(5)), int(rng.integers(3)))
            t = 0.0
            for _ in range(steps):
                a = (int(np.argmax(Q[s])) if name == "learned"
                     else int(rng.integers(N_ACTIONS)) if name == "random" else 0)
                s, r = step(s, a, rng)
                t += r
            totals.append(t)
        out[name] = float(np.mean(totals))
    return out


def load_or_train(force=False):
    """Load models/q_table.npy; train (about 10 s) if it does not exist."""
    if not force and Q_FILE.exists():
        Q = np.load(Q_FILE)
        if Q.shape == N_STATES_SHAPE + (N_ACTIONS,):
            return Q
    Q, log = train(verbose=False)
    MODELS_DIR.mkdir(exist_ok=True)
    np.save(Q_FILE, Q)
    log.to_csv(LOG_FILE, index=False)
    return Q


# -------------------------------------------------------------- recommendation
def bus_level_from_count(available, required_proxy):
    ratio = available / max(required_proxy, 1.0)
    return int(np.searchsorted([0.5, 0.85, 1.15, 1.6], ratio, side="right"))


def recommend(Q, activity_class, period, condition, available_buses, required_proxy,
              freq_level=1):
    """Advisory recommendation for one route / time period."""
    activity = ACTIVITY.index(activity_class)
    p = TIME_PERIODS.index(period)
    c = CONDITIONS.index(condition)
    b = bus_level_from_count(available_buses, required_proxy)
    state = (activity, p, c, b, freq_level)
    q = Q[state]
    best = int(np.argmax(q))
    immediate = [reward_for(state, a)[0] for a in range(N_ACTIONS)]
    need = service_need(activity, p, c)
    supply = b + freq_level - 1
    if supply < need:
        why = (f"Proxy service need ({need}) is above current supply ({supply}): "
               f"{activity_class.lower()} scheduled activity, {period}, {condition.lower()} conditions.")
    elif supply > need:
        why = (f"Current supply ({supply}) exceeds the proxy need ({need}) for "
               f"{activity_class.lower()} activity in {period} ({condition.lower()}); "
               "resources may be under-used.")
    else:
        why = (f"Current supply ({supply}) already matches the proxy need ({need}); "
               "keeping it avoids unnecessary change cost.")
    return {"state": {"activity": activity_class, "period": period, "condition": condition,
                      "bus_level": BUS_LEVELS[b], "frequency_level": FREQ_LEVELS[freq_level],
                      "available_buses": available_buses,
                      "required_buses_proxy": round(float(required_proxy), 2)},
            "need_level": need, "supply_level": supply,
            "q_values": dict(zip(ACTIONS, np.round(q, 3))),
            "immediate_rewards": dict(zip(ACTIONS, np.round(immediate, 2))),
            "recommended_action": ACTIONS[best],
            "expected_return": float(q[best]),
            "expected_immediate_reward": float(immediate[best]),
            "reason": why}


def lookup_route(route_short_name, period):
    if not ML_DATASET_FILE.exists():
        sys.exit("ERROR: run python preprocessing\\create_ml_dataset.py first.")
    df = pd.read_csv(ML_DATASET_FILE, dtype={"route_id": str, "route_short_name": str})
    row = df[(df["route_short_name"] == str(route_short_name)) & (df["time_period"] == period)]
    if row.empty:
        sys.exit(f"ERROR: no data for route '{route_short_name}' in period '{period}'. "
                 f"Periods: {TIME_PERIODS}")
    return row.iloc[0]


def print_recommendation(route, rec):
    s = rec["state"]
    print(f"\nRoute {route}")
    print(f"Time period: {s['period']}")
    print(f"Scheduled activity level: {s['activity']}   Condition (simulated input): {s['condition']}")
    print(f"Buses: {s['available_buses']} available vs ~{s['required_buses_proxy']} needed by the "
          f"schedule  -> fleet level '{s['bus_level']}', frequency '{s['frequency_level']}'")
    print("\nQ-values:")
    for a, v in rec["q_values"].items():
        print(f"  {a:<30} Q = {v:8.3f}   immediate reward = {rec['immediate_rewards'][a]:5.2f}")
    print(f"\nRecommendation:\n  {rec['recommended_action']}")
    print(f"Expected reward: {rec['expected_return']:.2f} (discounted return), "
          f"{rec['expected_immediate_reward']:.2f} (immediate)")
    print(f"Reason:\n  {rec['reason']}")
    print("\nNOTE: simulation-based, advisory decision support - not real-world bus control.")


def main():
    print("=" * 60)
    print("TransitMind - Q-Learning decision support (simulated environment)")
    print("=" * 60)
    MODELS_DIR.mkdir(exist_ok=True)
    print("Training Q-table (6000 episodes x 12 steps)...")
    Q, log = train()
    np.save(Q_FILE, Q)
    log["moving_avg_reward"] = log["total_reward"].rolling(100, min_periods=1).mean()
    log.round(4).to_csv(LOG_FILE, index=False)
    ev = evaluate(Q)
    print(f"\nMean episode reward over 500 simulated episodes: learned policy {ev['learned']:.1f} | "
          f"random {ev['random']:.1f} | always keep {ev['keep']:.1f}")
    print(f"Saved {Q_FILE}\nSaved {LOG_FILE}")

    route = sys.argv[1] if len(sys.argv) > 1 else "301"
    period = sys.argv[2] if len(sys.argv) > 2 else "Morning Peak"
    row = lookup_route(route, period)
    required = float(row["buses_required_proxy"])
    available = max(1, int(round(required * 0.7)))      # demo: fleet 30% under the schedule need
    rec = recommend(Q, row["activity_class"], period, "Congested", available, required)
    print_recommendation(route, rec)


if __name__ == "__main__":
    main()

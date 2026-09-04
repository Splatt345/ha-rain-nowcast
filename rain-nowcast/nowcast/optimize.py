"""Bounded, guarded coordinate-wise search over STEPS params. One run, logged,
auto-reverting anything that doesn't clearly improve. See README for the rule.
"""
import json
import os
import subprocess
import time

from backtest import BASELINE_CONFIG, load_stack, score_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "data/optimize_log.jsonl")
MAX_ITERS = 12
MAX_SECONDS = 20 * 60

# Each candidate perturbs ONE param away from current_best. ar_order and
# ar_context_frames are coupled (context = order + 1) and vary together.
CANDIDATES = [
    {"n_ens_members": 3},
    {"n_ens_members": 12},
    {"n_ens_members": 24},
    {"n_cascade_levels": 4},
    {"n_cascade_levels": 8},
    {"ar_order": 1, "ar_context_frames": 2},
    {"ar_order": 3, "ar_context_frames": 4},
    {"oflow_method": "vet"},
    {"oflow_method": "proesmans"},
    {"noise_method": "parametric"},
]


def log(entry):
    with open(LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def commit(msg):
    subprocess.run(["git", "add", "-A"], cwd=ROOT, check=True)
    subprocess.run(["git", "commit", "-q", "-m", msg], cwd=ROOT, check=True)


def main():
    R = load_stack()
    best_config = dict(BASELINE_CONFIG)
    best = {"mean_csi": 0.3965424126674087, "mean_rmse": 13.754147582340828}  # baseline, reused per directive

    start = time.time()
    n_accept = n_reject = 0

    for i, delta in enumerate(CANDIDATES):
        if i >= MAX_ITERS or time.time() - start > MAX_SECONDS:
            print(f"Stopping at iter {i}: budget reached.")
            break

        candidate_config = {**best_config, **delta}
        try:
            result = score_config(candidate_config, R=R)
        except Exception as e:
            log({"iter": i, "delta": delta, "error": str(e), "accepted": False})
            n_reject += 1
            print(f"iter {i} {delta}: ERROR {e} -> rejected")
            continue

        improved = result["mean_csi"] > best["mean_csi"] + 0.005
        not_worse_rmse = result["mean_rmse"] <= best["mean_rmse"] * 1.15
        accept = improved and not_worse_rmse

        log({
            "iter": i,
            "delta": delta,
            "mean_csi": result["mean_csi"],
            "mean_rmse": result["mean_rmse"],
            "prev_best_csi": best["mean_csi"],
            "prev_best_rmse": best["mean_rmse"],
            "accepted": accept,
        })

        if accept:
            prev_csi = best["mean_csi"]
            best_config = candidate_config
            best = result
            n_accept += 1
            commit(
                f"optimize: accept {delta} "
                f"(CSI {prev_csi:.4f}->{result['mean_csi']:.4f}, "
                f"RMSE->{result['mean_rmse']:.4f})"
            )
            print(f"iter {i} {delta}: ACCEPTED csi={result['mean_csi']:.4f} rmse={result['mean_rmse']:.4f}")
        else:
            n_reject += 1
            reason = "csi not improved enough" if not improved else "rmse regressed >15%"
            print(f"iter {i} {delta}: rejected ({reason}) csi={result['mean_csi']:.4f} rmse={result['mean_rmse']:.4f}")

    wall = time.time() - start

    # Held-out check on a different point in the same event's evolution.
    held_out = score_config(best_config, R=R, n_obs=20, n_lead=12)

    print("\n=== FINAL ===")
    print("best_config:", best_config)
    print("search-split:", {"mean_csi": best["mean_csi"], "mean_rmse": best["mean_rmse"]})
    print("held-out:", {"mean_csi": held_out["mean_csi"], "mean_rmse": held_out["mean_rmse"]})
    print(f"iterations: {min(len(CANDIDATES), MAX_ITERS)}  accepted: {n_accept}  rejected: {n_reject}  wall_s: {wall:.1f}")

    with open(os.path.join(ROOT, "data/optimize_final.json"), "w") as f:
        json.dump({
            "best_config": best_config,
            "search_split": {"mean_csi": best["mean_csi"], "mean_rmse": best["mean_rmse"]},
            "held_out_split": {"mean_csi": held_out["mean_csi"], "mean_rmse": held_out["mean_rmse"]},
            "n_accepted": n_accept,
            "n_rejected": n_reject,
            "wall_seconds": wall,
        }, f, indent=2)


if __name__ == "__main__":
    main()

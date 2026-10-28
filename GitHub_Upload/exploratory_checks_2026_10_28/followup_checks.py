"""Training-only boundary/constant benchmark and fixed-rule bootstrap checks."""

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import run_checks as checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    args = parser.parse_args()
    output = Path(__file__).resolve().parent / "results"
    original_cv = pd.read_csv(output / "ridge_training_cv.csv")
    summary = json.loads((output / "summary.json").read_text())
    frame = pd.read_csv(args.data)
    main_frame = frame[frame.segment.isin(["Mens E-Mail", "No E-Mail"])].copy()
    main_frame["treatment"] = (main_frame.segment == "Mens E-Mail").astype(int)
    train, remainder = checks.stratified_split(main_frame, 0.4)
    _, test = checks.stratified_split(remainder, 0.5)
    arm_frames = {"No E-Mail": train[train.treatment == 0], "Mens E-Mail": train[train.treatment == 1]}
    checks.ALPHAS = [1.0, 10000.0, 100000.0, 1000000.0]
    boundary_rows = []
    for offset, (name, subset) in enumerate(arm_frames.items()):
        rows, _, folds = checks.training_cv(subset, name, 20261028 + offset)
        for row in rows:
            row["model"] = "ridge"
            if row["alpha"] in [1.0, 10000.0]:
                old = original_cv[(original_cv.arm == name) & (original_cv.alpha == row["alpha"])].iloc[0]
                assert np.isclose(row["out_of_fold_mse"], old.out_of_fold_mse, rtol=1e-12)
        boundary_rows.extend(rows)
        y = subset.spend.to_numpy()
        squared_error = np.empty(len(subset))
        fold_mse = []
        fold_values = folds.cv_fold.to_numpy()
        for fold in range(5):
            held_out = fold_values == fold
            errors = (y[held_out] - y[~held_out].mean()) ** 2
            squared_error[held_out] = errors
            fold_mse.append(errors.mean())
        boundary_rows.append({"arm": name, "alpha": None, "out_of_fold_mse": float(squared_error.mean()), "fold_mse_sd": float(np.std(fold_mse, ddof=1)), "n": len(subset), "purchasers": int((y > 0).sum()), "model": "training-fold mean only"})
    pd.DataFrame(boundary_rows).to_csv(output / "training_only_boundary_check.csv", index=False)

    chosen = summary["selected_alpha"]
    original_score = checks.ridge_predict(arm_frames["Mens E-Mail"], test, 1) - checks.ridge_predict(arm_frames["No E-Mail"], test, 1)
    selected_score = checks.ridge_predict(arm_frames["Mens E-Mail"], test, chosen["Mens E-Mail"]) - checks.ridge_predict(arm_frames["No E-Mail"], test, chosen["No E-Mail"])
    k = int(0.25 * len(test))
    original = checks.selected_by_score(original_score, k)
    selected = checks.selected_by_score(selected_score, k)
    treatment, y = test.treatment.to_numpy(), test.spend.to_numpy()
    ipw = 2 * treatment * y - 2 * (1 - treatment) * y
    differences = np.column_stack([(selected.astype(float) - k / len(test)) * ipw * 1000, (selected.astype(float) - original.astype(float)) * ipw * 1000])
    intervals = checks.bootstrap_intervals(differences, treatment, draws=5000, seed=20261030)
    rows = []
    for col, label in enumerate(["Original-grid CV-selected minus expected random", "Original-grid CV-selected minus alpha=1"]):
        rows.append({"comparison": label, "estimate_per_1000": float(differences[:, col].mean()), "bootstrap_low": float(intervals[0, col]), "bootstrap_high": float(intervals[1, col]), "draws": 5000, "seed": 20261030})
    pd.DataFrame(rows).to_csv(output / "ridge_selected_bootstrap.csv", index=False)
    print(pd.DataFrame(boundary_rows).to_string(index=False))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

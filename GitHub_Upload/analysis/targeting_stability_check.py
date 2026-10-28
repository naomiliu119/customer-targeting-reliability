"""Independent checks and exploratory training stability analysis.

Usage: python targeting_stability_check.py /path/to/hillstrom.csv --output-dir results
Uses NumPy and pandas; does not change the source repositories.
"""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd


def approximate_mode(counts, n_draws, rng):
    continuous = counts / counts.sum() * n_draws
    floor = np.floor(continuous)
    needed = int(n_draws - floor.sum())
    for remainder in np.sort(np.unique(continuous - floor))[::-1]:
        if needed == 0:
            break
        indices = np.flatnonzero(continuous - floor == remainder)
        count = min(len(indices), needed)
        floor[rng.choice(indices, count, replace=False)] += 1
        needed -= count
    return floor.astype(int)


def stratified_split(frame, test_fraction):
    """Reproduce the repository's sklearn stratified split with seed 42."""
    labels = frame.treatment.to_numpy()
    counts = np.bincount(labels)
    n_test = int(np.ceil(test_fraction * len(frame)))
    rng = np.random.RandomState(42)
    train_counts = approximate_mode(counts, len(frame) - n_test, rng)
    test_counts = approximate_mode(counts - train_counts, n_test, rng)
    train, test = [], []
    for label in range(len(counts)):
        indices = np.flatnonzero(labels == label)[rng.permutation(counts[label])]
        train.extend(indices[: train_counts[label]])
        test.extend(indices[train_counts[label] : train_counts[label] + test_counts[label]])
    return frame.iloc[rng.permutation(train)].copy(), frame.iloc[rng.permutation(test)].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    checksum = hashlib.sha256(args.data.read_bytes()).hexdigest()
    expected = "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"
    if checksum != expected:
        raise ValueError("Dataset checksum differs from repository provenance.")

    frame = pd.read_csv(args.data)
    main_frame = frame[frame.segment.isin(["Mens E-Mail", "No E-Mail"])].copy()
    main_frame["treatment"] = (main_frame.segment == "Mens E-Mail").astype(int)
    train, remainder = stratified_split(main_frame, 0.4)
    validation, test = stratified_split(remainder, 0.5)
    numeric = ["recency", "history", "mens", "womens", "newbie"]
    categorical = ["channel", "zip_code"]
    categories = {column: sorted(train[column].unique()) for column in categorical}

    def encode(data):
        return np.column_stack(
            [data[numeric].to_numpy(float)]
            + [np.column_stack([(data[column].to_numpy() == value).astype(float)
                                for value in categories[column]]) for column in categorical]
        )

    x_train, x_test = encode(train), encode(test)
    y_train = train.spend.to_numpy()
    treatment = train.treatment.to_numpy()
    indices = [np.flatnonzero(treatment == label) for label in [0, 1]]

    def ridge_predictions(sample):
        x, target = x_train[sample].copy(), x_test.copy()
        means, scales = x[:, :5].mean(0), x[:, :5].std(0)
        scales[scales == 0] = 1
        x[:, :5] = (x[:, :5] - means) / scales
        target[:, :5] = (target[:, :5] - means) / scales
        x_mean, y_mean = x.mean(0), y_train[sample].mean()
        centered = x - x_mean
        coefficient = np.linalg.solve(
            centered.T @ centered + np.eye(x.shape[1]),
            centered.T @ (y_train[sample] - y_mean),
        )
        return (target - x_mean) @ coefficient + y_mean

    n_test = len(test)
    n_selected = int(0.25 * n_test)

    def select(score):
        selected = np.zeros(n_test, dtype=bool)
        selected[np.argsort(-score, kind="stable")[:n_selected]] = True
        return selected

    mu_email, mu_control = ridge_predictions(indices[1]), ridge_predictions(indices[0])
    original = select(mu_email - mu_control)
    historical = select(test.history.to_numpy())
    test_t, test_y = test.treatment.to_numpy(), test.spend.to_numpy()
    ipw = 2 * test_t * test_y - 2 * (1 - test_t) * test_y
    primary = (original - n_selected / n_test) * ipw * 1000
    primary_se = primary.std(ddof=1) / np.sqrt(n_test)
    z = NormalDist().inv_cdf(0.975)
    if not np.isclose(primary.mean(), 175.27043022456206, rtol=1e-9):
        raise AssertionError("Independent primary estimate does not match repository.")
    if int((original & historical).sum()) != 1288:
        raise AssertionError("Audience overlap does not match repository.")
    aipw = mu_email - mu_control + 2 * test_t * (test_y - mu_email) - 2 * (1 - test_t) * (test_y - mu_control)

    # NEW exploratory check: bootstrap training observations within treatment arms.
    # Refit both ridge models and numeric scaling. Score the unchanged test
    # covariates; test outcomes are NOT used in this audience-stability check.
    # Categorical levels stay fixed at those observed in the original training set.
    rng = np.random.default_rng(20261028)
    draws = 200
    frequency = np.zeros(n_test)
    retained = []
    for _ in range(draws):
        score = ridge_predictions(rng.choice(indices[1], len(indices[1]), replace=True)) - ridge_predictions(rng.choice(indices[0], len(indices[0]), replace=True))
        selected = select(score)
        frequency += selected
        retained.append(float((selected & original).sum() / n_selected))
    frequency /= draws

    result = {
        "dataset_sha256": checksum,
        "split_sizes": {"train": len(train), "validation": len(validation), "test": n_test},
        "independent_checks": {
            "primary_advantage_per_1000": float(primary.mean()),
            "primary_approximate_95pct_interval": [float(primary.mean() - z * primary_se), float(primary.mean() + z * primary_se)],
            "selected_per_rule": n_selected,
            "overlap_incremental_historical": int((original & historical).sum()),
            "replaced_per_rule": int((original & ~historical).sum()),
            "purchasers_in_test": int((test_y > 0).sum()),
            "purchasers_in_disagreement_set": int(((original ^ historical) & (test_y > 0)).sum()),
            "aipw_incremental_minus_historical_per_1000": float(((original.astype(int) - historical.astype(int)) * aipw * 1000).mean()),
        },
        "new_exploratory_training_stability": {
            "status": "Exploratory sensitivity analysis added after original results were known; not independent confirmation",
            "draws": draws,
            "seed": 20261028,
            "retained_original_audience_fraction_quantiles_10_50_90": np.quantile(retained, [0.1, 0.5, 0.9]).tolist(),
            "original_selected_selection_frequency_quantiles_10_50_90": np.quantile(frequency[original], [0.1, 0.5, 0.9]).tolist(),
            "original_selected_with_frequency_below_half": int((frequency[original] < 0.5).sum()),
            "all_customers_with_frequency_between_10pct_and_90pct": int(((frequency > 0.1) & (frequency < 0.9)).sum()),
            "interpretation": "Training sensitivity of the specified ridge targeting rule on fixed test covariates; not a revenue confidence interval or external validation",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "targeting_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    pd.DataFrame({"source_row": test.index, "original_selected": original, "selection_frequency": frequency}).to_csv(args.output_dir / "selection_frequencies.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

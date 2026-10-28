"""Exploratory ridge sensitivity and fixed-audience email-version matching.

Run with Python, NumPy and pandas. See Analysis_Plan.md for decisions fixed
before this run. This script writes only to its own results directory.
"""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import NormalDist
import sys

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from targeting_stability_check import stratified_split

ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
NUMERIC = ["recency", "history", "mens", "womens", "newbie"]
CATEGORICAL = ["channel", "zip_code"]
Z = NormalDist().inv_cdf(0.975)
EXPECTED_SHA = "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"


def encoded_pair(training, evaluation):
    means = training[NUMERIC].to_numpy(float).mean(0)
    scales = training[NUMERIC].to_numpy(float).std(0)
    scales[scales == 0] = 1.0
    levels = {c: sorted(training[c].unique()) for c in CATEGORICAL}

    def transform(frame):
        blocks = [(frame[NUMERIC].to_numpy(float) - means) / scales]
        for col in CATEGORICAL:
            blocks.append(np.column_stack([(frame[col].to_numpy() == v).astype(float) for v in levels[col]]))
        return np.column_stack(blocks)

    return transform(training), transform(evaluation)


def ridge_predict(training, evaluation, alpha):
    x, xe = encoded_pair(training, evaluation)
    y = training.spend.to_numpy(float)
    xm, ym = x.mean(0), y.mean()
    xc = x - xm
    beta = np.linalg.solve(xc.T @ xc + alpha * np.eye(x.shape[1]), xc.T @ (y - ym))
    return (xe - xm) @ beta + ym


def training_cv(frame, arm, seed):
    rng = np.random.default_rng(seed)
    fold_ids = np.empty(len(frame), dtype=int)
    buyers = frame.spend.to_numpy() > 0
    for label in [False, True]:
        shuffled = rng.permutation(np.flatnonzero(buyers == label))
        for fold, subset in enumerate(np.array_split(shuffled, 5)):
            fold_ids[subset] = fold
    rows = []
    for alpha in ALPHAS:
        squared_errors = np.empty(len(frame))
        fold_mse = []
        for fold in range(5):
            held_out = fold_ids == fold
            prediction = ridge_predict(frame.iloc[np.flatnonzero(~held_out)], frame.iloc[np.flatnonzero(held_out)], alpha)
            errors = (frame.spend.to_numpy()[held_out] - prediction) ** 2
            squared_errors[held_out] = errors
            fold_mse.append(float(errors.mean()))
        rows.append({"arm": arm, "alpha": alpha, "out_of_fold_mse": float(squared_errors.mean()), "fold_mse_sd": float(np.std(fold_mse, ddof=1)), "n": len(frame), "purchasers": int(buyers.sum())})
    best = min(rows, key=lambda row: row["out_of_fold_mse"])["alpha"]
    indices = pd.DataFrame({"source_row": frame.index, "arm": arm, "cv_fold": fold_ids})
    return rows, best, indices


def selected_by_score(score, n_to_select):
    selected = np.zeros(len(score), dtype=bool)
    selected[np.argsort(-np.asarray(score), kind="stable")[:n_to_select]] = True
    return selected


def paired_summary(values, strata=None):
    values = np.asarray(values, dtype=float)
    if strata is None:
        se2 = values.var(ddof=1) / len(values)
    else:
        strata = np.asarray(strata)
        se2 = sum((np.mean(strata == h) ** 2) * values[strata == h].var(ddof=1) / np.sum(strata == h) for h in np.unique(strata))
    se = np.sqrt(se2)
    mean = values.mean()
    return {"estimate": float(mean), "se": float(se), "ci_low": float(mean - Z * se), "ci_high": float(mean + Z * se)}


def bootstrap_intervals(values, strata, draws=5000, seed=20261029):
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    groups = [np.flatnonzero(strata == h) for h in np.unique(strata)]
    rng = np.random.default_rng(seed)
    estimates = np.zeros((draws, values.shape[1]))
    for draw in range(draws):
        total = np.zeros(values.shape[1])
        for group in groups:
            total += values[rng.choice(group, len(group), replace=True)].sum(axis=0)
        estimates[draw] = total / len(values)
    return np.quantile(estimates, [0.025, 0.975], axis=0)


def history_group(frame):
    return np.select([(frame.mens == 1) & (frame.womens == 0), (frame.mens == 0) & (frame.womens == 1), (frame.mens == 1) & (frame.womens == 1)], ["Men's only", "Women's only", "Both"], default="Unexpected")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(args.data.read_bytes()).hexdigest()
    assert sha == EXPECTED_SHA, "Data checksum mismatch"
    frame = pd.read_csv(args.data)
    main_frame = frame[frame.segment.isin(["Mens E-Mail", "No E-Mail"])].copy()
    main_frame["treatment"] = (main_frame.segment == "Mens E-Mail").astype(int)
    train, rem = stratified_split(main_frame, 0.4)
    validation, test = stratified_split(rem, 0.5)
    assert [len(train), len(validation), len(test)] == [25567, 8523, 8523]
    arm_frames = {"No E-Mail": train[train.treatment == 0], "Mens E-Mail": train[train.treatment == 1]}
    cv_rows, chosen, folds = [], {}, []
    for offset, (name, subset) in enumerate(arm_frames.items()):
        rows, best, fold_indices = training_cv(subset, name, seed=20261028 + offset)
        cv_rows.extend(rows)
        chosen[name] = best
        folds.append(fold_indices)
    pd.DataFrame(cv_rows).to_csv(out / "ridge_training_cv.csv", index=False)
    pd.concat(folds, ignore_index=True).to_csv(out / "ridge_cv_fold_assignments.csv", index=False)
    print("Training-only CV selections:", chosen, flush=True)

    settings = [(f"alpha_{alpha:g}", alpha, alpha) for alpha in ALPHAS]
    settings.append(("training_CV_selected", chosen["Mens E-Mail"], chosen["No E-Mail"]))
    sensitivity_rows = []
    for partition, evaluation in [("validation", validation), ("test", test)]:
        y = evaluation.spend.to_numpy()
        t = evaluation.treatment.to_numpy()
        score = 2 * t * y - 2 * (1 - t) * y
        original_score = ridge_predict(arm_frames["Mens E-Mail"], evaluation, 1) - ridge_predict(arm_frames["No E-Mail"], evaluation, 1)
        for name, a_email, a_control in settings:
            mu_email = ridge_predict(arm_frames["Mens E-Mail"], evaluation, a_email)
            mu_control = ridge_predict(arm_frames["No E-Mail"], evaluation, a_control)
            for contact_rate in [0.1, 0.25, 0.5]:
                k = int(contact_rate * len(evaluation))
                q = k / len(evaluation)
                original = selected_by_score(original_score, k)
                selected = selected_by_score(mu_email - mu_control, k)
                historical = selected_by_score(evaluation.history.to_numpy(), k)
                gain = selected * score * 1000
                versus_random = paired_summary((selected.astype(float) - q) * score * 1000)
                versus_original = paired_summary((selected.astype(float) - original.astype(float)) * score * 1000)
                versus_history = paired_summary((selected.astype(float) - historical.astype(float)) * score * 1000)
                sensitivity_rows.append({"partition": partition, "setting": name, "alpha_email": a_email, "alpha_control": a_control, "contact_rate": contact_rate, "selected": k, "additional_revenue_per_1000": float(gain.mean()), "advantage_over_random": versus_random["estimate"], "advantage_ci_low": versus_random["ci_low"], "advantage_ci_high": versus_random["ci_high"], "difference_from_original": versus_original["estimate"], "difference_from_original_ci_low": versus_original["ci_low"], "difference_from_original_ci_high": versus_original["ci_high"], "difference_from_history": versus_history["estimate"], "difference_from_history_ci_low": versus_history["ci_low"], "difference_from_history_ci_high": versus_history["ci_high"], "retained_original_audience": float((selected & original).sum() / k)})
                if partition == "test" and name == "alpha_1" and contact_rate == 0.25:
                    assert np.isclose(versus_random["estimate"], 175.27043022456206, rtol=1e-9)
                    assert int((selected & historical).sum()) == 1288
    pd.DataFrame(sensitivity_rows).to_csv(out / "ridge_sensitivity.csv", index=False)

    # Extend the original partitions without reassigning any original men's/control row.
    women = frame[frame.segment == "Womens E-Mail"].copy()
    women["treatment"] = 0
    women_train, women_rem = stratified_split(women, 0.4)
    women_valid, women_test = stratified_split(women_rem, 0.5)
    partitions = {"train": pd.concat([train, women_train]), "validation": pd.concat([validation, women_valid]), "test": pd.concat([test, women_test])}
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert set(partitions[left].index).isdisjoint(partitions[right].index)
    assert sum(map(len, partitions.values())) == 64000
    all_indices = pd.concat([pd.DataFrame({"source_row": p.index, "partition": name, "assignment": p.segment.to_numpy()}) for name, p in partitions.items()], ignore_index=True)
    all_indices.to_csv(out / "extended_split_assignments.csv", index=False)
    evaluation = partitions["test"].copy()
    evaluation["history_group"] = history_group(evaluation)
    assert "Unexpected" not in set(evaluation.history_group)
    group_order = ["Men's only", "Women's only", "Both"]
    cell_rows, cells = [], {}
    for group in group_order:
        for arm in ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]:
            outcomes = evaluation.loc[(evaluation.history_group == group) & (evaluation.segment == arm), "spend"].to_numpy()
            cells[group, arm] = outcomes
            cell_rows.append({"history_group": group, "assignment": arm, "n": len(outcomes), "purchasers": int((outcomes > 0).sum()), "mean_spend": float(outcomes.mean()), "spend_variance": float(outcomes.var(ddof=1))})
    pd.DataFrame(cell_rows).to_csv(out / "history_assignment_cells.csv", index=False)
    effect_rows = []
    for group in group_order:
        for left, right in [("Mens E-Mail", "No E-Mail"), ("Womens E-Mail", "No E-Mail"), ("Mens E-Mail", "Womens E-Mail")]:
            a, b = cells[group, left], cells[group, right]
            difference = float(a.mean() - b.mean())
            se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
            effect_rows.append({"history_group": group, "comparison": f"{left} minus {right}", "revenue_difference_per_customer": difference, "se": se, "ci_low": difference - Z * se, "ci_high": difference + Z * se})
    pd.DataFrame(effect_rows).to_csv(out / "history_email_effects.csv", index=False)
    interaction_cells = [("Men's only", "Mens E-Mail", 1), ("Men's only", "Womens E-Mail", -1), ("Women's only", "Mens E-Mail", -1), ("Women's only", "Womens E-Mail", 1)]
    interaction = sum(sign * cells[group, arm].mean() for group, arm, sign in interaction_cells)
    interaction_var = sum(cells[group, arm].var(ddof=1) / len(cells[group, arm]) for group, arm, _ in interaction_cells)
    interaction_se = np.sqrt(interaction_var)

    # All version policies share exactly the same historical-spending audience.
    order = np.lexsort((evaluation.index.to_numpy(), -evaluation.history.to_numpy()))
    k = int(0.25 * len(evaluation))
    audience = np.zeros(len(evaluation), dtype=bool)
    audience[order[:k]] = True
    groups = evaluation.history_group.to_numpy()
    matching_men = np.where(groups == "Men's only", 1.0, np.where(groups == "Women's only", 0.0, 0.5))
    weights = {"All men's email": np.ones(len(evaluation)), "All women's email": np.zeros(len(evaluation)), "Expected 50/50 versions": np.full(len(evaluation), 0.5), "History-matched versions": matching_men}
    labels = evaluation.segment.to_numpy()
    y = evaluation.spend.to_numpy()
    nominal = {arm: 1 / 3 for arm in np.unique(labels)}
    realized = {arm: float(np.mean(labels == arm)) for arm in np.unique(labels)}
    policy_rows, paired_rows = [], []
    base = None
    for estimator, probabilities in [("nominal_IPW", nominal), ("arm_normalized_IPW", realized)]:
        contributions = {}
        for name, w_men in weights.items():
            w_women = 1 - w_men
            assert np.allclose(w_men + w_women, 1)
            assert np.all((w_men >= 0) & (w_men <= 1))
            row_values = audience * y * ((labels == "Mens E-Mail") * w_men / probabilities["Mens E-Mail"] + (labels == "Womens E-Mail") * w_women / probabilities["Womens E-Mail"] - (labels == "No E-Mail") / probabilities["No E-Mail"]) * 1000
            contributions[name] = row_values
            summary = paired_summary(row_values, labels)
            policy_rows.append({"estimator": estimator, "policy": name, "selected": k, **summary})
        for benchmark in ["All men's email", "All women's email", "Expected 50/50 versions"]:
            values = contributions["History-matched versions"] - contributions[benchmark]
            summary = paired_summary(values, labels)
            paired_rows.append({"estimator": estimator, "comparison": "History matched minus " + benchmark, **summary})
            # Since audiences are identical, control outcomes cancel in paired version differences.
            assert np.allclose(values[labels == "No E-Mail"], 0)
        if estimator == "nominal_IPW":
            assert np.allclose(contributions["Expected 50/50 versions"], (contributions["All men's email"] + contributions["All women's email"]) / 2)
            names = list(weights)
            comparison_names = ["History matched minus " + b for b in names[:3]]
            matrix = np.column_stack([contributions[n] for n in names] + [contributions[names[3]] - contributions[n] for n in names[:3]])
            intervals = bootstrap_intervals(matrix, labels)
            for row, low, high in zip(policy_rows[-4:], intervals[0, :4], intervals[1, :4]):
                row.update({"bootstrap_low": float(low), "bootstrap_high": float(high)})
            for row, low, high in zip(paired_rows[-3:], intervals[0, 4:], intervals[1, 4:]):
                row.update({"bootstrap_low": float(low), "bootstrap_high": float(high)})
            base = {name: float(contributions[name].mean()) for name in names}
    pd.DataFrame(policy_rows).to_csv(out / "message_policy_values.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(out / "message_policy_comparisons.csv", index=False)
    pd.DataFrame({"source_row": evaluation.index, "history_group": groups, "selected": audience, "mens_email_probability_if_selected": matching_men}).to_csv(out / "matching_policy_assignments.csv", index=False)

    summary = {
        "status": "Exploratory checks after original test outcomes were known; no source files or main manuscript modified",
        "dataset_sha256": sha,
        "alpha_grid": ALPHAS,
        "cv_selection_criterion": "pooled out-of-fold spending MSE, separately by training treatment arm; no validation/test outcomes in selection",
        "selected_alpha": chosen,
        "cv_fold_seeds": {"No E-Mail": 20261028, "Mens E-Mail": 20261029},
        "extended_split_sizes": {name: len(p) for name, p in partitions.items()},
        "extended_test_arm_counts": evaluation.segment.value_counts().to_dict(),
        "fixed_matching_audience_selected": k,
        "fixed_matching_audience_history_counts": evaluation.loc[audience, "history_group"].value_counts().to_dict(),
        "interaction_definition": "(men's minus women's version in men's-only history) minus (men's minus women's version in women's-only history)",
        "interaction_revenue_per_customer": {"estimate": float(interaction), "se": float(interaction_se), "ci_low": float(interaction - Z * interaction_se), "ci_high": float(interaction + Z * interaction_se)},
        "nominal_matching_policy_values_per_1000": base,
        "bootstrap": {"resamples": 5000, "strata": "randomized email assignment", "seed": 20261029, "fixed": "history-based audience and all message rules"},
        "limitations": ["Pointwise exploratory intervals; no multiplicity adjustment", "No independent new campaign; original test outcomes previously examined", "CV criterion is spending prediction, not policy revenue", "Matching compares message versions within a common historical-spending audience", "No profits, customer gender, send-time experiment, or isolated message-component effects"],
        "environment": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

# Exploratory checks for the writing sample

Written before running the checks below, October 28, 2026. The original test results have already been examined. This document is not a preregistration, and none of these checks creates a new untouched confirmatory evaluation.

## Aim

Understand whether the existing results depend on an arbitrary ridge setting and examine Naomi's question about matching email version to merchandise-purchase history. Preserve the original analysis, the existing notebook, and the main manuscript. Save code, results, interpretation, and proposed manuscript wording in this separate directory.

## A. Ridge settings

- Reproduce the existing men's-email/no-email train, validation, and test split and alpha=1 baseline before adding alternatives.
- Compare alpha values 0.1, 1, 10, 100, 1,000, and 10,000. This is a modest sensitivity grid, not an exhaustive optimization.
- Use five-fold cross-validation within each treatment arm's original training data. Balance folds by observed purchase/nonpurchase status. Fit standardization and categorical encoding only to each fold's training observations.
- Select the alpha with the smallest pooled out-of-fold spending mean squared error separately for email and control. This optimizes a prediction criterion, not policy revenue or profit.
- Evaluate the fixed-alpha rules and the training-CV-selected pair at 10%, 25%, and 50% contact rates. Keep historical spending and the original rule as benchmarks.
- Report validation and test estimates, paired approximate intervals, audience overlap with the original rule, and the CV scores. Do not select a winner based on test revenue or change this grid after seeing it.

## B. Purchase-history matching

- Retain the original split of the men's-email/no-email observations. Independently split the women's-email arm 60/20/20 with seed 42 and combine the respective partitions. Record all indices. These are exploratory partitions of existing data, not newly collected observations.
- Define the three observed history groups using pre-campaign indicators: men's merchandise only, women's merchandise only, and both. Do not interpret these indicators as customer gender.
- Describe group-specific revenue and purchase counts for men's email, women's email, and no email on the combined evaluation partition. Report differences between each email and control and between the two email versions, with approximate intervals.
- Estimate one explicit interaction: (men's minus women's email effect in men's-only history) minus (men's minus women's email effect in women's-only history). This tests effect differences directly; differing subgroup significance labels are not an interaction test.
- For a policy comparison at nominal 25% contact rate, hold the audience identical across email-version policies: the highest prior-year spending customers. This isolates assignment of message version within a fixed audience.
- Compare: men's email for everyone selected; women's email for everyone selected; an expected 50/50 email-version mix; and history matching (men's-only -> men's email, women's-only -> women's email, both -> expected 50/50 mix).
- Use nominal randomized assignment probabilities of one-third. Add a check normalized by realized evaluation-arm proportions to show any effect of slight count imbalances.
- Estimate additional revenue relative to contacting nobody and paired differences between the fixed policies. Report treatment-stratified approximate and 5,000-resample percentile intervals. All intervals are exploratory and pointwise, with no multiplicity adjustment.
- Do not change the behavior for customers with both histories or choose a different audience after inspecting results.

## Interpretation rules

Keep unfavorable results. A change in alpha is not evidence of improved targeting unless revenue evidence supports that claim. Matching policies that share a fixed audience do not optimize whom to contact. The 25% contact limit is a scenario, not an estimated optimum. Revenue is not profit; costs and margins are absent. Message versions are bundled treatments; no isolated claim about images, wording, gender, timing, or psychological mechanisms is supported.

## Reproducibility

Use the publisher's dataset with SHA-256 `0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece`. Seeds and outputs must be saved. Numerical checks must include the published baseline estimate, audience overlap, split separation, policy weights, and a direct subgroup-effect calculation.

## Follow-up after the first run

The six-value grid selected its largest alpha (10,000) for both arms. This is a boundary result, not evidence that the optimal penalty has been found. Before any further checks, specify this limited follow-up: compare alpha=1, 10,000, 100,000, and 1,000,000 with an intercept-only training-fold mean using the same training-only folds. Do not use this follow-up to replace the already reported original-grid model or search for more favorable test results. Do not evaluate newly added penalties on test data in this follow-up.

Also compute treatment-stratified percentile bootstrap intervals for the already fixed original-grid CV-selected rule at the 25% contact rate, comparing it with random targeting and alpha=1. This checks the approximate intervals for a sparse revenue outcome; it does not reselect the model. Use 5,000 resamples, seed 20261030, with fitted models and selection labels fixed.

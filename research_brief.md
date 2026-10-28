# Evaluating Customer Targeting Under Sparse Purchasing Outcomes

Naomi Liu | Independent research project

## Research question

At a fixed contact rate, does targeting customers by estimated incremental revenue improve revenue relative to random targeting?

The distinction matters because customers likely to purchase after receiving an email may also purchase without it. Predicting purchasing behavior and identifying customers whose behavior changes because of marketing are different tasks.

## Data and design

I analyzed Kevin Hillstrom’s public MineThatData email experiment, which contains 64,000 customers assigned to men’s merchandise email, women’s merchandise email, or no email. This project is a secondary analysis of the original experiment.

The primary analysis compares men’s merchandise email with no email, retaining 42,613 customers, including those who spent nothing. I divided these observations into training (60%), validation (20%), and final test (20%) samples, stratified by treatment assignment. Predictors included prior spending, purchase recency, merchandise purchase history, customer status, purchasing channel, and geographic category. Post-campaign outcomes were excluded from model inputs.

## Methods

I compared four targeting rules: historical spending, predicted purchase probability, predicted revenue under email, and estimated incremental revenue.

Purchase probabilities were estimated using logistic regression fitted to emailed training customers. Separate ridge regressions estimated spending under email and no email; their prediction difference supplied the incremental-revenue score.

Each rule ranked customers using historical characteristics. The primary contact rate was 25%, with 10% and 50% examined as secondary comparisons. Models were fitted using training data and held fixed during evaluation.

Using the experiment’s assignment probabilities, I estimated each rule’s additional revenue relative to contacting nobody and its advantage over expected random targeting at the same contact rate. Results are expressed per 1,000 eligible customers. Paired approximate confidence intervals were supplemented by a stratified bootstrap for the primary comparison.

## Findings

In the training sample, email assignment increased estimated revenue by $0.670 per customer, with a 95% confidence interval of $0.305 to $1.034. A positive average campaign effect, however, does not establish that a targeting model can identify the customers with the largest incremental response.

At the primary 25% contact rate, incremental-revenue targeting selected 2,130 final test customers. Its estimated additional revenue was $341.84 per 1,000 eligible customers, compared with $166.57 under random targeting. The estimated advantage was therefore $175.27, with an approximate 95% confidence interval of −$72.45 to $422.99. The bootstrap interval was −$51.45 to $447.32.

Both intervals include zero. The final test therefore does not establish that incremental-revenue targeting outperforms random targeting. All twelve final test comparisons across rules and contact rates also had pointwise intervals that included zero.

The incremental-revenue rule’s estimated advantage was negative in validation and positive in final testing. This difference illustrates the uncertainty surrounding performance estimates in these samples.

## Exploratory extension: Comparing targeting decisions directly

After examining the original test results, I compared incremental-revenue targeting directly with historical-spending and predicted-revenue targeting at the fixed 25% contact rate. I retained the original fitted models and selected audiences, evaluating paired revenue differences using both inverse-probability and augmented inverse-probability estimators.

Under the augmented estimator, incremental targeting’s estimated revenue difference was −$52.69 per 1,000 eligible customers relative to historical spending (approximate 95% CI: −$342.26 to $236.88) and −$7.41 relative to predicted revenue (−$314.27 to $299.45). Both evaluation methods and their stratified bootstrap sensitivity checks produced intervals containing zero. Adjustment changed the estimates modestly and provided little improvement in precision.

The rules nevertheless selected substantially different audiences. Incremental and historical-spending targeting shared 1,288 of 2,130 selections; each selected 842 customers excluded by the other. Incremental-only customers had lower mean prior spending ($190.01 versus $485.75), less recent purchases, and more frequent women’s merchandise purchase history (94.54% versus 43.11%).

This extension documents a substantial reallocation of contacts without establishing an accompanying revenue benefit. Customer profiles are descriptive, comparisons are exploratory, and uncertainty calculations hold fitted models and selection labels fixed.

## Additional exploratory analyses

These checks were developed after the original validation and test results were examined. They extend the same experiment and do not provide independent confirmation.

### Sensitivity to training observations

I refitted the two original ridge models across 200 bootstrap resamples drawn within the training assignment groups. Each refit selected 2,130 customers using the same test characteristics, without using test spending to choose recipients. The median retained share of the original audience was 70.8%. Of the original 2,130 selections, 524 were selected in fewer than half of the refits. These summaries measure changes in allocation; they are not revenue confidence intervals and do not establish that changed selections are costly.

### Ridge-setting sensitivity

Five-fold training-only cross-validation selected alpha=10,000 for both spending models from the original sensitivity grid. Spending prediction error improved by only about 0.03% for the no-email model and 0.05% for the email model. The selected rule retained 81.8% of the original audience at the main contact rate. Its estimated test advantage over the original rule was $43.04 per 1,000 eligible customers, with an approximate 95% interval of -$177.82 to $263.91. This does not establish a targeting-revenue improvement. A training-only follow-up checked larger penalties and a constant benchmark; those alternatives were not evaluated on test outcomes.

### Matching email version to purchase history

A separate extension uses all three assignment groups and fixes the audience at the 3,200 highest-spending customers in a combined evaluation sample of 12,801. The matching rule assigns the corresponding version to single-category buyers and an expected 50/50 mix to customers with both histories. It compares this rule with common men's, common women's, and mixed versions for the same audience. Every paired interval includes zero. The results do not establish an advantage from the specified matching rule, nor do they establish equivalence. Merchandise-purchase history does not identify gender.

The [second notebook](notebooks/02_exploratory_checks.ipynb) reviews the saved outputs. The [analysis guide](analysis/README.md) documents the estimators, execution order, and verification scope. Scripts, exported results, and figures accompany both analyses.

## Limitations and implications

Purchasing was rare: only 578 of the original 64,000 customers purchased. Revenue estimates consequently depend on relatively few observed transactions. The models also showed limited predictive performance in validation.

The uncertainty calculations hold fitted models and selection labels fixed; they do not capture uncertainty from repeating the entire model-fitting and targeting procedure. Secondary comparisons are exploratory and use pointwise intervals. The analysis covers one historical campaign. Its main comparison uses men’s email versus no email; the later message-version extension adds the women’s-email group. Revenue is not profit because campaign costs and contribution margins are unavailable.

These findings support a cautious conclusion: a positive average marketing effect does not, by itself, demonstrate a reliable advantage from customer targeting. This study provides an evaluation of specified targeting rules in one public experiment, with uncertainty reported alongside estimated gains.

## Reproducibility

The repository includes the analysis notebook, package versions, data-source instructions, dataset checksum, exported results, and comparison figure. A session restart and complete notebook rerun reproduced the reported primary results.

The later scripts and saved results are separate additions. Notebook 02 reviews saved outputs by default; it does not refit models unless the optional regeneration mode is enabled. The independent verification scope and its limits are recorded in `analysis/README.md`.

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

## Limitations and implications

Purchasing was rare: only 578 of the original 64,000 customers purchased. Revenue estimates consequently depend on relatively few observed transactions. The models also showed limited predictive performance in validation.

The uncertainty calculations hold fitted models and selection labels fixed; they do not capture uncertainty from repeating the entire model-fitting and targeting procedure. Secondary comparisons are exploratory and use pointwise intervals. The analysis covers one campaign and the men’s-email versus no-email contrast. Revenue is not profit because campaign costs and contribution margins are unavailable.

These findings support a cautious conclusion: a positive average marketing effect does not, by itself, demonstrate a reliable advantage from customer targeting. This study provides an evaluation of specified targeting rules in one public experiment, with uncertainty reported alongside estimated gains.

## Reproducibility

The repository includes the analysis notebook, package versions, data-source instructions, dataset checksum, exported results, and comparison figure. A session restart and complete notebook rerun reproduced the reported primary results.

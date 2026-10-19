# Customer Targeting and Revenue Uncertainty

An independent analysis of the public Hillstrom email experiment examining whether customer-targeting models improve incremental revenue over random selection.

## Research question

At the same contact rate, does targeting customers by estimated incremental revenue outperform random targeting and simple predictive benchmarks?

## Data

The original dataset contains 64,000 customers randomly assigned to a men’s merchandise email, a women’s merchandise email, or no email. Outcomes were measured over two weeks.

The primary analysis compares men’s email with no email, using 42,613 customers. The original experiment and data collection were conducted by the dataset provider.

Source: [Kevin Hillstrom’s MineThatData Email Analytics Challenge](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html).

## Research design

- Training: 25,567 customers; validation: 8,523; final test: 8,523.
- Treatment-stratified 60/20/20 split with random seed 42.
- Inputs: prior spending, purchase recency, merchandise-purchase history, new-customer status, purchasing channel, and location category.
- Standardization and categorical encoding fitted within training pipelines.
- Primary contact rate: 25%; secondary rates: 10% and 50%.
- Final test comparisons fixed after validation and before examining test outcomes.

## Targeting strategies

1. **Historical spending:** rank customers by prior-year spending.
2. **Purchase probability:** logistic regression fitted to emailed training customers.
3. **Predicted revenue:** ridge regression fitted to emailed training customers.
4. **Incremental revenue:** difference between separate email and control ridge-regression predictions.
5. **Random targeting:** expected performance of random selection at the same contact rate.

Ridge regressions use alpha = 1. All spending analyses retain nonpurchasers and their zero spending.

## Evaluation

Policy gains are estimated using inverse-probability weighting and the known experimental assignment probabilities. Results are expressed as additional revenue per 1,000 eligible customers relative to emailing nobody.

Comparisons with random targeting use paired approximate 95% confidence intervals. The primary comparison also includes a 5,000-resample stratified bootstrap sensitivity check, holding fitted models and selection labels fixed.

Method reference: [GRF guide to policy evaluation](https://grf-labs.github.io/grf/articles/maq.html).

## Main findings

At the primary 25% contact rate, final-test estimated additional revenue was:

| Strategy | Revenue per 1,000 eligible customers |
|---|---:|
| Random targeting | $166.57 |
| Incremental-revenue targeting | $341.84 |

The estimated advantage was **$175.27**, with an approximate 95% confidence interval of **−$72.45 to $422.99**. The bootstrap interval was **−$51.45 to $447.32**.

Both intervals include zero. The held-out test therefore does not establish an advantage over random targeting.

All secondary final-test comparisons also had pointwise intervals spanning zero. The incremental rule’s estimated advantage changed direction between validation and final testing.

## Limitations

Purchases are rare: only 578 customers purchased in the original dataset. Revenue estimates can therefore depend substantially on a small number of transactions.

The analysis covers one historical experiment, a two-week outcome window, and a limited set of models. Confidence intervals describe evaluation uncertainty for fixed models; they do not capture the full uncertainty from model training. Costs are unavailable, so results concern revenue rather than profit.

## Repository contents

- `notebooks/01_targeting_analysis.ipynb`: data checks, modeling, evaluation, and interpretation.
- `results/`: comparison tables, primary test summary, and figure.

## Running the analysis

Open the notebook in Google Colab, run cells in order, and upload the original CSV when prompted. The analysis uses pandas, NumPy, SciPy, scikit-learn, and Matplotlib.

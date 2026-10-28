# Customer Targeting and Revenue Uncertainty

Independent analysis by Naomi Liu of Kevin Hillstrom's public randomized email experiment.

**Question:** At the same contact budget, do learned targeting rules generate more additional revenue than random selection or simple benchmarks?

## Findings

The original analysis compares historical spending, predicted purchase probability, predicted spending, and estimated incremental revenue. The primary comparison uses men's-merchandise email versus no email, with 42,613 customers split into training, validation, and test samples.

At the nominal 25% contact rate, incremental-revenue targeting's estimated advantage over expected random targeting is **$175.27 per 1,000 eligible customers**, with an approximate 95% interval of **-$72.45 to $422.99**. The corresponding bootstrap interval is **-$51.45 to $447.32**. These results leave the relative revenue benefit unresolved.

Later exploratory checks examine:

- **Training sensitivity:** across 200 training resamples, the median retained share of the original audience is 70.8%. These are selection-stability summaries, not revenue confidence intervals.
- **Ridge settings:** training-only cross-validation produces very small prediction-error improvements without establishing a targeting-revenue benefit.
- **Message matching:** assigning email versions using prior merchandise purchases shows no clear benefit over common versions within the same fixed audience.

The later checks were developed after the original test outcomes were examined. They extend the same experiment and do not provide independent confirmation. Revenue is not profit; costs and margins are unavailable.

## Where to start

| File or folder | Purpose |
|---|---|
| [Notebook 01](notebooks/01_targeting_analysis.ipynb) | Original data checks, models, and evaluation |
| [Notebook 02](notebooks/02_exploratory_checks.ipynb) | Later exploratory results and figures |
| [Research brief](research_brief.md) | Research question, design, findings, and limits |
| [Analysis scripts and instructions](analysis/README.md) | Reproduce the later checks |
| [Original results](results/) / [Later results](results/exploratory/) | Exported tables and figures |
| [Data instructions](data/README.md) | Public source and dataset checksum |

## Run the notebooks

[Open notebook 01 in Colab](https://colab.research.google.com/github/naomiliu119/customer-targeting-reliability/blob/main/notebooks/01_targeting_analysis.ipynb) | [Open notebook 02 in Colab](https://colab.research.google.com/github/naomiliu119/customer-targeting-reliability/blob/main/notebooks/02_exploratory_checks.ipynb)

Notebook 01 uses the recorded environment in `requirements.txt`. In Colab, upload that file, install it with `%pip install -r /content/requirements.txt`, restart the session, and run the notebook in order. Upload the verified public CSV when prompted. The original reference environment uses Python 3.13.16.

Notebook 02 reviews saved outputs by default. Its first Colab cell downloads this repository when supporting files are absent. To rerun the later checks, supply the verified CSV and enable `RECOMPUTE`; the [analysis guide](analysis/README.md) gives the commands and verification scope.

## Data

The source experiment assigned 64,000 previous purchasers to men's-merchandise email, women's-merchandise email, or no email, with two-week outcomes. Merchandise labels do not identify gender. Nonpurchasers remain in the revenue analyses with zero spending. No employer records are used.

[Original experiment and data description](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html)

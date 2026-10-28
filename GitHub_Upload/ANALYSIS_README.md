# Analysis and reproducibility guide

Updated October 28, 2026. This guide covers the original customer-targeting workflow and the later exploratory checks.

## Contents and provenance

- `notebooks/01_targeting_analysis.ipynb`, `data/README.md`, `requirements.txt`, and `results/`: the original workflow, data provenance, recorded environment, and exported results. The later additions preserve these original files.
- `analysis/targeting_stability_check.py` and `analysis/results/`: independent baseline checks and the later exploratory training-resampling analysis. The validation diagnostic record identifies which values were independently reproduced and which were checked against notebook output.
- `exploratory_checks_2026_10_28/`: the later analysis plan, ridge-setting and message-matching scripts, results, and interpretation. These checks were developed after the original test results were examined.
- `analysis/make_paper_figures.py`: generates the two analysis figures from saved outputs and checks the principal displayed numbers. It does not refit models.
- `analysis/audit_paper_figures.py`: independently refits the original models and checks the revenue figure, audience membership, and purchase-model validation metrics.
- `notebooks/02_exploratory_checks.ipynb`: a companion notebook with executed review cells for the later results, figures, and optional regeneration commands. It reviews saved outputs by default and preserves the original notebook's separate role.
- `figures/`: the two figures in PNG, PDF, and SVG formats.

Project repository: https://github.com/naomiliu119/customer-targeting-reliability. These companion materials distinguish the original workflow from the later exploratory analyses. No employer records are used.

## Dataset

Download the publisher's CSV using the source link in `data/README.md`. The raw CSV is not included. Preserve its row order. The required SHA-256 checksum is:

`0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece`

Original experiment and dictionary: https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html.

## Execution order

1. For the original four-rule analysis, open `notebooks/01_targeting_analysis.ipynb` in Google Colab, upload the verified CSV when prompted, and run the cells in order. `requirements.txt` records the original package versions. The notebook uses Colab-specific upload/export calls.
2. From the repository root, run the independent check and exploratory training resampling:

```sh
python analysis/targeting_stability_check.py /path/to/hillstrom.csv --output-dir analysis/results
```

3. Run the initial later checks, followed by the documented follow-up:

```sh
python exploratory_checks_2026_10_28/run_checks.py /path/to/hillstrom.csv
python exploratory_checks_2026_10_28/followup_checks.py /path/to/hillstrom.csv
```

These scripts require NumPy and pandas. The follow-up reads the initial check's results and writes to that same results directory. It does not test newly added ridge penalties on test outcomes.

4. To regenerate the two analysis figures, run:

```sh
python analysis/make_paper_figures.py
```

This reads the original tables copied to `analysis/results/original_notebook/` and the saved training-resampling outputs. It requires NumPy and Matplotlib. The current figures were rendered with Matplotlib 3.11.2; this rendering version does not replace the original notebook's recorded statistical environment.

The optional independent figure audit is:

```sh
python analysis/audit_paper_figures.py /path/to/hillstrom.csv
```

It requires NumPy, pandas, SciPy, and scikit-learn. The saved audit records the actual package versions used. Open notebook 02 for a readable review of the results; its default `RECOMPUTE=False` mode checks saved outputs rather than refitting models.

For Colab, notebook 02's setup cell downloads this public repository when supporting files are absent. Its review cells display the two saved figure files directly. For a local Jupyter session, download or clone the complete repository and open the notebook from its `notebooks/` directory. The saved review cells were executed locally; the Colab download path has not been tested in a live Colab runtime.

## Verification scope

The original purchase and spending models were independently refitted for the figure audit. All four revenue estimates and normal intervals at the 25% contact rate matched the original exports, as did the audience overlap and purchase-model validation metrics. The complete 200-refit training exercise was rerun, and all 8,523 saved selection frequencies and stability summaries matched exactly. The augmented comparison with historical spending and email-spending validation diagnostics were also independently reproduced in the earlier review. The initial ridge-grid and message-matching script was rerun and its outputs compared with the saved results. The training-only boundary and selected-model bootstrap follow-up was checked against its code and saved outputs, without an independent rerun. The original percentile bootstrap intervals were not independently regenerated. The full original notebook is provided so that remaining step can be rerun in its documented environment.

The revenue intervals concern fixed fitted rules. The separate training-resampling exercise concerns audience sensitivity. The later extensions do not create a new untouched confirmation sample.


## Estimators and interpretation

The main comparison includes the men's-merchandise email and no-email groups. Spending includes zeros. With email indicator T, observed spending Y, and known conditional assignment probability p=1/2, each customer's IPW contribution is z = T*Y/p - (1-T)*Y/(1-p). A fixed selection rule s has estimated additional revenue 1000*mean(s*z) per 1,000 eligible customers. At quota q, expected random targeting has value 1000*q*mean(z). Direct rule differences use the same customers and outcomes: 1000*mean((s-r)*z).

The paired normal intervals use the sample standard deviation of the customer-level difference contributions divided by the square root of the evaluation sample size, with a multiplier of 1.96. Bootstrap checks resample within assignment groups while keeping fitted scores and selections fixed. These pointwise intervals do not adjust for multiple comparisons. They do not represent the uncertainty from repeating the full model-development procedure.

The augmented direct comparisons replace z with mu1-mu0 + T*(Y-mu1)/p - (1-T)*(Y-mu0)/(1-p), using spending predictions from the original training-only models. The training-resampling exercise separately refits the ridge models across 200 stratified training resamples and records changes in selected audiences; it is an allocation-sensitivity analysis.

The three-arm message comparison holds the selected audience fixed. A selected customer's contribution is 1000*Y*[3*I(men)*wM + 3*I(women)*wW - 3*I(no email)], where wM and wW are the policy's probabilities of choosing each version and sum to one. Policy differences share the same control contribution. The analysis-plan file and scripts document the fixed audience, resampling, and training-only ridge checks.

Interpretation relies on the provider's reported random assignment, positive assignment probabilities, consistency of the assigned email versions, and no effects of one customer's assignment on another customer's spending. Rules use pre-campaign information and training-only fitted models. These comparisons concern assignment to email, not opening it, and do not establish transferability to another campaign. Campaign costs and margins are unavailable, so the outcomes are revenue rather than profit.

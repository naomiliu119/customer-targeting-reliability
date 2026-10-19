# Dataset provenance

## Source

Kevin Hillstrom’s **MineThatData E-Mail Analytics and Data Mining Challenge**, published March 20, 2008.

- [Original challenge and data dictionary](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html)
- [Publisher’s original CSV download](http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv)

The dataset was downloaded for this project on October 18, 2026 and saved locally as `hillstrom.csv`.

## File verification

- **Rows:** 64,000
- **Columns:** 12
- **SHA-256:** `0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece`

The checksum was verified against the CSV uploaded to the analysis notebook.

## Experimental design

The publisher describes random assignment to three groups:

| Assignment | Customers |
|---|---:|
| Mens E-Mail | 21,307 |
| Womens E-Mail | 21,387 |
| No E-Mail | 21,306 |

The email labels describe the merchandise featured, not customer gender. Outcomes cover the two weeks following campaign assignment.

## Variables used

Pre-campaign predictors are `recency`, `history`, `mens`, `womens`, `newbie`, `channel`, and `zip_code`.

`segment` records experimental assignment. `visit`, `conversion`, and `spend` are post-assignment outcomes and are excluded from predictor inputs.

`history_segment` categorizes prior spending and is omitted from modeling because the underlying `history` amount is available.

The primary analysis uses men’s email versus no email: **42,613 customers**. Nonpurchasers and zero spending are retained.

## Reproduction

Download the original CSV and verify its SHA-256 fingerprint before analysis. Preserve its row order because the seeded sample split depends on that order.

Open `notebooks/01_targeting_analysis.ipynb` in Google Colab and upload the CSV when prompted. Run the remaining cells in order.

This project analyzes an existing public experiment; the project author did not conduct the original randomization or data collection.

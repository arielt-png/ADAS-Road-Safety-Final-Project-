# Data

The original datasets used in this study are not distributed through this public repository.

The analysis is based on aggregated vehicle and road-safety data derived from the Israeli Central Bureau of Statistics (CBS) SOFI dataset.

To reproduce the analysis, users must obtain the required source data separately and place the analysis-ready CSV file in this directory.

The primary analysis script expects the following filename:

`sofi_stratum_year.csv`

The script also recognizes:

`sofi_stratum_year(1).csv`
`sofi_stratum_year(2).csv`

The analysis-ready dataset must contain the variables required by `ADAS_complete_analysis.py`.

The repository's `.gitignore` file is configured to prevent CSV, Excel, SPSS, and Stata data files placed in this directory from being uploaded to GitHub.

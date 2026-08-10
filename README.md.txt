# ADAS and Road Safety in Israel

## Overview

This repository contains the Python analysis code accompanying the M.Sc. final project examining the association between vehicle technology, Advanced Driver Assistance Systems (ADAS), and road traffic accident rates in Israel.

The analysis evaluates differences in exposure-adjusted accident rates across vehicle production cohorts and examines the association between measured vehicle safety features and accident frequency and severity.

## Authors

Etay Rabino
Ariel Trink

M.Sc. Systems Engineering and Technology Management
Holon Institute of Technology (HIT)

## Data

The primary analysis is based on aggregated vehicle and road-safety data derived from the Israeli Central Bureau of Statistics (CBS) SOFI dataset.

The original research data are **not included in this public repository**. Users wishing to reproduce the analysis must obtain the required source data separately and place the analysis-ready CSV file in the `/data` directory.

The primary analysis script expects the following filename:

`sofi_stratum_year.csv`

Alternative filenames recognized by the script include:

`sofi_stratum_year(1).csv`
`sofi_stratum_year(2).csv`

## Analysis

The repository contains the code used for the empirical analyses reported in the final project, including:

* Data validation and preparation
* Vehicle-age and production-cohort construction
* Descriptive statistics
* Exposure-adjusted accident-rate calculations
* Vehicle production-cohort Poisson regression
* Quasi-Poisson-style robustness inference
* Negative Binomial (NB2) robustness analysis
* Calendar-year fixed-effects analysis
* Alternative sample-definition sensitivity checks
* Accident-severity analysis
* Individual ADAS feature analysis
* Active and warning/support feature-count analysis
* Cohort attenuation analysis
* Feature-level severity analysis
* Engine-volume sensitivity analysis
* Generation of analysis tables and figures

## Primary Analytical Specification

The primary cohort analysis:

* Includes vehicles aged 1–20 years
* Uses production cohorts from 2000 onward
* Uses the 2000–2004 production cohort as the reference category
* Models accident-event counts using Poisson regression
* Uses the logarithm of total kilometers driven as an exposure offset
* Adjusts for vehicle-age category and fuel type

The dedicated ADAS feature analysis focuses on production cohorts from 2010 onward, using the 2010–2014 cohort as the reference category.

## Repository Structure

```text
ADAS-Road-Safety-Thesis/
│
├── analysis/
│   └── ADAS_complete_analysis.py
│
├── data/
│   └── README.md
│
├── outputs/
│
├── .gitignore
├── README.md
└── requirements.txt
```

### `analysis/`

Contains the Python analysis script used to reproduce the statistical analyses, tables, and figures.

### `data/`

Location for the analysis-ready SOFI dataset. The underlying research dataset is not distributed through this repository.

### `outputs/`

The analysis script automatically creates subdirectories containing generated tables, model results, diagnostics, and figures.

## Running the Analysis

From the repository root, run:

```bash
python analysis/ADAS_complete_analysis.py
```

The script will automatically search for:

```text
data/sofi_stratum_year.csv
```

and will save generated results under:

```text
outputs/ADAS_complete_analysis_outputs/
```

Alternatively, explicit paths can be supplied:

```bash
python analysis/ADAS_complete_analysis.py \
    --data "data/sofi_stratum_year.csv" \
    --output "outputs/ADAS_complete_analysis_outputs"
```

## Software Requirements

The analysis was implemented in Python.

Required Python packages include:

* numpy
* pandas
* scipy
* matplotlib
* statsmodels
* patsy

Exact package requirements are provided in `requirements.txt`.

## Reproducibility

The repository is intended to document the analytical workflow used in the final project and allow the reported analyses to be reproduced by researchers who have legitimate access to the underlying source data.

Because the original CBS-derived research data are not distributed here, complete reproduction requires obtaining the relevant source data separately.

## Interpretation

The analyses are observational and use aggregated stratum-year data. Estimated production-cohort and vehicle-feature associations should therefore not be interpreted as clean causal estimates of the effects of ADAS technology.

## Citation

When referring to this analysis code, please cite the accompanying M.Sc. final project.

A permanent archived version and citation information may be added upon final thesis submission.

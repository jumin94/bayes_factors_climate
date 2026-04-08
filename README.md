# Bayes Factors for Climate Science: Quantifying Evidence for Scientific Hypotheses
Julia Mindlin & Marlene Kretschmer | Leipzig Institute for Meteorology, Leipzig University, Leipzig, Germany

## Overview
This repository contains the full set of Python examples accompanying the article...
All examples are written for **Python 3.11** and run in a reproducible environment provided in `environment.yml`.

---

Clone the repository:

```bash
git clone https://github.com/jumin94/bayes_factors_climate.git
cd bayes_factors_climate


conda env create -f environment.yml
conda activate climate-bayes


Repo structure

bayes_factors_climate/
├── Example1/ # Nested model analysis (linear vs quadratic trends)
├── Example2/ # Breakpoint models (global warming hiatus)
├── Example3/ # Correlation (ENSO --> GMST)
├── Example4/ # Multiple driver analysis (ENSO and IOD)
├── Example5/ # Record-shattering extremes analysis
├── Example6/ # Dynamical storyline evaluation using ESMValTool
├── conceptual_fig.ipynb # Conceptual illustration figure
├── climate_bayes_env.yaml # Conda environment file
└── README.md

All the codes are Jupyter Notebooks except for Example 6 (deterministic models). 
With the ESMValTool recipe  the code is downloaded and the diagnostic produces figure.


![Conceptual illustration of statistical hypotheses recurrently posed in climate science and covered in this work. This includes the (1) comparison of statistical models for a trend, (2) evaluation of the break-point hypothesis for trends, (3) significance maps for correlations, (4) testing multiple drivers as predictors, (5) comparison of the likelihood of record-breaking extremes under different models for trends, and (6) comparison of storylines against observations.](Figure1.png)

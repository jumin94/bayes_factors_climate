# Bayes Factors for Climate Science: Quantifying Evidence for Scientific Hypotheses
### Examples and Environment
### Julia Mindlin & Marlene Kretschmer

## Overview
This repository contains the full set of Python examples accompanying the analysis of teleconnections, Bayesian evidence quantification, partial regression, and storyline-based likelihoods.  
All examples are written for **Python 3.11** and run in a reproducible environment provided in `environment.yml`.

---

## 1. Installing the Environment

### Using Conda or Mamba (recommended)
Clone the repository:

```bash
git clone https://github.com/jumin94/bayes_factors_climate.git
cd bayes_factors_climate


conda env create -f environment.yml
conda activate climate-bayes


Repo structure

bayes_factors_climate/
├── Example1/ # Nested model analysis (linear vs quadratic trends)
├── Example2/ # TBD
├── Example3/ # Breakpoint models (global warming hiatus)
├── Example4/ # Multiple driver analysis (ENSO and IOD)
├── Example5/ # Record-shattering extremes analysis
├── Example6/ # Dynamical storyline evaluation using ESMValTool
├── conceptual_fig.ipynb # Conceptual illustration figure
├── climate_bayes_env.yaml # Conda environment file
└── README.md

All the codes are Jupyter Notebooks except for Example 6 (deterministic models). With the ESMValTool recipe 
the code is downloaded and the diagnostic produces the figure




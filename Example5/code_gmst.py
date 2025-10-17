import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from esmvaltool.diag_scripts.shared import run_diagnostic, group_metadata
import numpy as np

def load_observations():
    path = "/home/jmindlin/BF_codes/data/DCENT_GMST_statistics.txt"
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith("Year,"))

    df = pd.read_csv(
        path,
        skiprows=start,
        engine="python",
        sep=r"\s*,\s*",
    )

    rename_map = {
        "Year": "year",
        "GMST (°C)": "gmst",
        "1 s.d. (°C)": "gmst_sd",
        "GMST non infilled (°C)": "gmst_noninf",
        "1 s.d. (°C; non  infilled)": "gmst_noninf_sd",
    }
    df.columns = [re.sub(r"\s+", " ", c).strip() for c in df.columns]
    rename_map_norm = {re.sub(r"\s+", " ", k).strip(): v for k, v in rename_map.items()}
    df = df.rename(columns=rename_map_norm)

    for c in ["year", "gmst", "gmst_sd", "gmst_noninf", "gmst_noninf_sd"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    ds = xr.Dataset(
        {
            "GMST": (["year"], df["gmst"].values),
            "GMST_sd": (["year"], df["gmst_sd"].values),
        },
        coords={"year": df["year"].values.astype(int)},
    )
    return ds.sel(year=slice(1951,2024)) - ds.sel(year=slice(1951,1980)).mean(dim='year')



def bayes_factor_likelihood_ratio(y_obs, model_a, model_b, sigma):
    """
    Compute Bayes Factor (likelihood ratio) comparing two deterministic models to observations.
    Parameters:
    -----------
    y_obs : array-like
        Observed values (e.g., GMST time series).
    model_a : array-like
        Model A predictions (same length as y_obs).
    model_b : array-like
        Model B predictions (same length as y_obs).
    sigma : float
        Known standard deviation of observational uncertainty.
    Returns:
    --------
    float
        Bayes Factor (Model A over Model B)
    """
    y_obs = np.asarray(y_obs)
    model_a = np.asarray(model_a)
    model_b = np.asarray(model_b)
    sse_a = np.sum((y_obs - model_a)**2)
    sse_b = np.sum((y_obs - model_b)**2)
    bf = np.exp(-0.5 * (sse_a - sse_b) / sigma**2)
    return bf

def main(cfg):
    input_data = group_metadata(cfg['input_data'].values(), 'dataset')
    obs_ds = load_observations()
    obs_gmst = obs_ds["GMST"]

    model_means = []
    model_names = []

    for dataset_name, members in input_data.items():
        all_members = []
        for meta in members:
            ds = xr.open_dataset(meta['filename'])
            tas = ds[meta['short_name']].squeeze()
            tas_ann = tas.groupby("time.year").mean("time")
            tas_ann = tas_ann - tas_ann.sel(year=slice(1951, 1980)).mean()
            all_members.append(tas_ann)

        # Stack ensemble members for this dataset
        ens_stack = xr.concat(all_members, dim="member")
        model_mean = ens_stack.mean(dim="member")
        model_means.append(model_mean)
        model_names.append(dataset_name)

    # === Compute MME (multi-model mean) ===
    mme_stack = xr.concat(model_means, dim="dataset")
    mme_mean = mme_stack.mean(dim="dataset")
   
    # === Compute Bayes Factors per model ===
    bf_values = []

    for model_mean in model_means:
        bf = bayes_factor_likelihood_ratio(obs_gmst.values,model_mean.values,    mme_mean.values , sigma=obs_gmst.sel(year=slice('1950','1979')).std(dim='year').values)
        bf_values.append(1/bf)

    # # === Plot obs GMST time series as well as individual model means (gray) and MME (black) ===
    # fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    # for model_mean in model_means:
    #     model_mean.plot(ax=ax, color='gray', alpha=0.5, label=None)
    
    # ### Plot with colors and labels the models with BF < 1
    # for i, bf in enumerate(bf_values):
    #     if bf < 1:
    #         model_means[i].plot(ax=ax, linewidth=2, label=model_names[i]+' BF:'+str(round(1/bf,3)),alpha=0.8)

    # obs_gmst.plot(ax=ax, color='r', linewidth=2, label="Observations")
    # mme_mean.plot(ax=ax, color='k', linewidth=2, label="Multi-Model Ensemble Mean")
    # ax.set_title("Global Mean Surface Temperature Anomalies")
    # ax.set_ylabel("GMST Anomaly (°C)")
    # ax.legend()
    # ax.grid()
    # plt.tight_layout()
    # output_dir = cfg["plot_dir"]
    # os.makedirs(output_dir, exist_ok=True)
    # fig.savefig(os.path.join(output_dir, f"gmst_comparison_all_models.png"))

    # # === Plot Bayes Factors ===
    # fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    # bars = ax.bar(model_names, bf_values, color='steelblue')

    # for thresh, label in [(1, 'No preference'), (3, 'Substantial'), (10, 'Strong'), (30, 'Very strong')]:
    #     ax.axhline(thresh, linestyle='--', color='gray', linewidth=0.8)
    #     ax.text(len(model_names)-0.5, thresh + 0.2, label, va='bottom', ha='right', fontsize=8, color='gray')

    # ax.set_ylabel("Bayes Factor vs. MME (H₀)")
    # ax.set_title("Model vs. Multi-Model Ensemble")
    # ax.set_yscale("log")
    # ax.set_ylim(0, 100)
    # ax.grid(True, axis='y', which='both', linestyle=':', linewidth=0.5)

    # plt.xticks(rotation=90)
    # plt.tight_layout()

    # output_dir = cfg["plot_dir"]
    # os.makedirs(output_dir, exist_ok=True)
    # fig.savefig(os.path.join(output_dir, f"bayes_factors_vs_mme.png"))

    n_models = len(model_names)
    # === Generate distinct matplotlib default colors ===
    colors = plt.cm.tab10.colors  # tab10 has 10 distinct colors
    color_array = [colors[i % len(colors)] for i in range(n_models)]
    # === Create Two-Panel Plot ===
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), dpi=150, gridspec_kw={'height_ratios': [2, 1]})
    # --- Panel A: GMST Time Series ---
    ax = axes[0]
    for model_mean in model_means:
        model_mean.plot(ax=ax, color='gray', alpha=0.5, label=None)

    for i, bf in enumerate(bf_values):
        if bf < 1:
            model_means[i].plot(ax=ax, linewidth=2, label=f'{model_names[i]} (1/BF={1/bf:.1f})', color=color_array[i])

    obs_gmst.plot(ax=ax, color='r', linewidth=2, label="Observations")
    mme_mean.plot(ax=ax, color='k', linewidth=2, label="MME Mean")

    ax.set_title("a) Global Mean Surface Temperature Anomalies")
    ax.set_ylabel("GMST Anomaly (°C)")
    ax.legend()
    ax.grid()

    # --- Panel B: Bayes Factors with dual scales ---
    ax = axes[1]
    bar_colors = ['gray' if bf >= 1 else color_array[i] for i, bf in enumerate(bf_values)]
    bars = ax.bar(model_names, bf_values, color=bar_colors)
    # Left y-axis (linear scale)
    for thresh, label in [(1, 'No preference'), (3, 'Substantial'), (10, 'Strong'), (30, 'Very strong')]:
        ax.axhline(thresh, linestyle='--', color='gray', linewidth=0.8)
        ax.text(len(model_names)-0.5, thresh + 0.2, label, va='bottom', ha='right', fontsize=8, color='gray')

    ax.set_ylabel("Bayes Factor vs. MME (H₀)")
    ax.set_title("b) Model Comparison via Bayes Factors")
    ax.set_yscale("log")
    ax.set_ylim(0.01, 100)
    ax.grid(True, axis='y', which='both', linestyle=':', linewidth=0.5)
    ax.set_xticks(np.arange(len(model_names)))
    ax.set_xticklabels(model_names, rotation=90)
    # Right y-axis (log₁₀ BF)
    ax2 = ax.twinx()
    log_BFs = np.log10(bf_values)
    ax2.bar(model_names, log_BFs, width=0.3, color='salmon', align='edge')

    ax2.set_ylabel("log₁₀(Bayes Factor)", color='salmon')
    ax2.tick_params(axis='y', labelcolor='salmon')
    ax2.axhline(0, color='gray', linestyle='--')
    # Add Jeffreys' log scale
    # for thresh, label in [(np.log10(3), "Substantial"), (np.log10(10), "Strong"), 
    #                     (np.log10(30), "Very Strong"), (np.log10(100), "Decisive"),
    #                     (np.log10(1/3), "Substantial"), (np.log10(1/10), "Strong"), 
    #                     (np.log10(1/30), "Very Strong"), (np.log10(1/100), "Decisive")]:
    #     ax2.axhline(thresh, color='gray', linestyle=':', alpha=0.6)
    #     ax2.text(len(model_names) - 0.5, thresh, label, va='bottom', fontsize=8, color='gray')

    plt.tight_layout()
    output_dir = cfg["plot_dir"]
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"bayes_factors_vs_mme_new.png"))

if __name__ == "__main__":
    with run_diagnostic() as config:
        main(config)

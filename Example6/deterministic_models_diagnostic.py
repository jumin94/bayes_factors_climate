### General imports
import os
import glob
import json 
import netCDF4
import random
import xarray as xr
import numpy as np
import pandas as pd
import scipy.stats as stats
import urllib.request

### ESMValTool imports
from esmvaltool.diag_scripts.shared import run_diagnostic, get_cfg, group_metadata

### Ploting
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as mticker
from matplotlib.ticker import LogLocator, LogFormatterSciNotation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
plt.rcParams.update({
    "font.size": 18,          # base size (everything scales from this)
})


def bayes_factor_RD(obs,sl,slmean):
    try:
        RSS0 = np.sum((obs.values - slmean.values)**2)
        RSS1 = np.sum((obs.values - sl.values)**2)
    except:
        RSS0 = np.sum((obs.values - slmean.values[:-1])**2)
        RSS1 = np.sum((obs.values - sl.values[:-1])**2)

    # Compute log-likelihood for Model 1
    n = len(obs.values)
    logL1 = -n/2 * np.log(RSS0/3) - n/2 * np.log(2*np.pi) - n/2
    # Compute BIC for Model 1 - tiene un parametro por lo tanto k1 = 2
    k1 = 2
    BIC1 = k1 * np.log(n) - 2 * logL1

    # Compute log-likelihood for Model 2 -
    logL2 = -n/2 * np.log(RSS1/3) - n/2 * np.log(2*np.pi) - n/2
    # Compute BIC for Model 1  tiene un parametro por lo tanto k1 = 2
    k2 = 2
    BIC2 = k2 * np.log(n) - 2 * logL2

    # Compute the log of the Bayes Factor
    log_BF21 = 0.5 * (BIC1 - BIC2)

    # Compute the Bayes Factor
    BF_MEM_SL = np.exp(log_BF21)
    return(BF_MEM_SL)


def seasonal_data(data,season='DJF'):
    # select DJF
    DA_DJF = data.sel(time = data.time.dt.season==season)

    # calculate mean per year
    DA_DJF = DA_DJF.groupby(DA_DJF.time.dt.year).mean("time")
    DA_DJF = DA_DJF.rename({'year':'time'})
    return DA_DJF

def seasonal_data_months(data, months):
    """
    Selects specified months from an xarray object and averages the data for those months within each year.
    
    Parameters:
    - data: xarray.DataArray or xarray.Dataset
        The input data to process. It should have a 'time' coordinate.
    - months: list of int
        The months to select for averaging (1 = January, 2 = February, ..., 12 = December).
    
    Returns:
    - xarray.DataArray or xarray.Dataset
        The averaged data for the selected months within each year.
    """
    # Ensure 'time' coordinate is in a format that supports .dt accessor
    if np.issubdtype(data['time'].dtype, np.datetime64):
        time_coord = data['time']
    else:
        time_coord = xr.cftime_range(start=data['time'][0].values, periods=data['time'].size, freq='M')
        data = data.assign_coords(time=time_coord)

    # Select specified months
    selected_months_data = data.sel(time=data['time'].dt.month.isin(months))
    
    # Group by year and average the selected months within each year
    averaged_data = selected_months_data.groupby('time.year').mean(dim='time')
    
    return averaged_data.rename({'year':'time'})

def make_xarr(data,time):
    time_series = xr.DataArray(
    data,
    coords=[time],
    dims=["time"],
    name="time_array")
    return time_series
    

# def plot_bayes_panel(ax, bf_values, title):

#     bf_values = np.array(bf_values)
#     xpos = np.arange(len(bf_values))

#     # --- Evidence regions ---
#     regions = [
#         (1, 3, "negligible", 0.98),
#         (3, 10, "substantial", 0.7),
#         (10, 100, "strong", 0.5),
#         (100, 1e5, "decisive", 0.4)
#     ]

#     for y0, y1, _, alpha in regions:
#         ax.axhspan(y0, y1, color='gray', alpha=alpha, zorder=0)

#     # --- Bars ---
#     bars = ax.bar(xpos, bf_values, color="#6A3D9A",
#                   edgecolor='black', linewidth=0.6, zorder=3)

#     # Hatching for BF < 3
#     for i, bf in enumerate(bf_values):
#         if bf < 3:
#             bars[i].set_facecolor("white")
#             bars[i].set_hatch("//")

#     # --- Threshold lines ---
#     for t in [1, 3, 10, 100]:
#         ax.axhline(t, color='black', linestyle='--', linewidth=0.8)

#     # --- Axis ---
#     ax.set_yscale("log")
#     ax.set_ylim(0.0001, max(bf_values) * 1.3)

#     ax.set_xticks(xpos)
#     ax.set_xticklabels(["high", "low"])

#     ax.set_ylabel("Bayes factor (BF$_{10}$)")
#     ax.set_title(title, fontsize=16, loc='left')

#     # --- Annotate values ---
#     for i, bf in enumerate(bf_values):
#         y_pos = bf * 1.15 if bf >= 1 else bf * 2.5
#         ax.text(xpos[i], y_pos, f"{bf:.2g}", ha='center', fontsize=12)

#     # --- Clean ---
#     ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)

from matplotlib.ticker import LogLocator, LogFormatterMathtext

def plot_bayes_panel(ax, bf_values, title):

    bf_values = np.array(bf_values)
    xpos = np.arange(len(bf_values))

    # ======================
    # Symmetric evidence regions (log-space)
    # ======================
    regions = [
        (1, 3.2, "negligible", 0.95),
        (3.2, 10, "substantial", 0.75),
        (10, 100, "strong", 0.55),
        (100, 1000, "decisive", 0.1),
    ]

    # Upper (BF > 1)
    for y0, y1, _, alpha in regions:
        ax.axhspan(y0, y1, color='gray', alpha=alpha, zorder=0)

    # Lower (BF < 1) → mirrored
    for y0, y1, _, alpha in regions:
        ax.axhspan(1/y1, 1/y0, color='gray', alpha=alpha, zorder=0)

    # ======================
    # Bars anchored at 1
    # ======================
    heights = bf_values - 1  # positive or negative

    bars = ax.bar(
        xpos,
        heights,
        bottom=1,
        color="#6A3D9A",
        edgecolor='black',
        linewidth=0.6,
        zorder=3
    )

    # Hatching / color for weak evidence (|BF| < 3)
    for i, bf in enumerate(bf_values):
        if 1/3 < bf < 3:
            bars[i].set_facecolor("white")
            bars[i].set_hatch("//")

    # ======================
    # Threshold lines (both sides)
    # ======================
    thresholds = [1, 3.2, 10, 100]
    for t in thresholds:
        ax.axhline(t, color='black', linestyle='--', linewidth=0.8, zorder=2)
        if t != 1:
            ax.axhline(1/t, color='black', linestyle='--', linewidth=0.8, zorder=2)

    # ======================
    # Log scale
    # ======================
    ax.set_yscale("log")
    ax.set_ylim(1/100000, 100000)

    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))

    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)*0.1))
    ax.yaxis.set_minor_formatter(lambda *args: "")

    # ======================
    # Labels
    # ======================
    ax.set_xticks(xpos)
    ax.set_xticklabels(["high", "low"])

    ax.set_ylabel("Bayes factor (BF$_{10}$)")
    ax.set_title(title, fontsize=16, loc='left')

    # ======================
    # Annotate values
    # ======================
    for i, bf in enumerate(bf_values):
        if bf >= 1:
            y_pos = bf * 1.15
        else:
            y_pos = bf / 1.5
        ax.text(xpos[i], y_pos, f"{bf:.2g}", ha='center', fontsize=12)

    # ======================
    # Clean look
    # ======================
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_mean_with_shading(ax, data, variable_name, data_source, title,
                          obs, sl_high_high_l, sl_high_low_l,
                          subplot_title, time, time_obs, time_sl, time_sl_long):

    # --- Ensemble stats ---
    mean_values = data.mean(dim='model')
    max_values = data.max(dim='model')
    min_values = data.min(dim='model')

    # --- Plot time series ---
    ax.plot(time_obs, obs.values, color='green', linewidth=2)
    ax.plot(time, mean_values.values, color='black')
    ax.fill_between(time, min_values.values, max_values.values,
                    color='grey', alpha=0.3)

    ax.plot(time_sl_long, sl_high_high_l.values, color='red')
    ax.plot(time_sl_long, sl_high_low_l.values, color='blue')

    # --- Titles & labels ---
    ax.set_title(subplot_title, loc='left')
    ax.set_xlabel('Year')
    ax.set_ylabel(variable_name)
    ax.tick_params(axis='both')

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ======================
    # LEGEND TEXT PER PANEL
    # ======================
    if subplot_title == 'a)':
        red_label = "high global warming"
        blue_label = "low global warming"
    elif subplot_title == 'b)':
        red_label = "high tropical warming"
        blue_label = "low tropical warming"
    elif subplot_title == 'c)':
        red_label = "high polar vortex strengthening"
        blue_label = "low polar vortex strengthening"
    else:
        red_label = "high"
        blue_label = "low"

    # --- Custom legend ---
    legend_elements = [
        Patch(facecolor='grey', alpha=0.3, label='model ensemble'),
        Line2D([0], [0], color='black', lw=2, label='model ensemble mean'),
        Line2D([0], [0], color='red', lw=2, label=red_label),
        Line2D([0], [0], color='blue', lw=2, label=blue_label),
        Line2D([0], [0], color='green', lw=2, label='observations'),
    ]

    ax.legend(handles=legend_elements,
              fontsize=12,
              frameon=False,
              loc='upper left')

    # --- Compute Bayes factors ---
    BF_high = bayes_factor_RD(
        obs.sel(time=slice('1960', '2022')),
        make_xarr(sl_high_high_l.values,
                  mean_values.sel(time=slice('1950', '2099')).time).sel(time=slice('1960', '2022')),
        mean_values.sel(time=slice('1960', '2022'))
    )

    BF_low = bayes_factor_RD(
        obs.sel(time=slice('1960', '2022')),
        make_xarr(sl_high_low_l.values,
                  mean_values.sel(time=slice('1950', '2099')).time).sel(time=slice('1960', '2022')),
        mean_values.sel(time=slice('1960', '2022'))
    )

    return float(BF_high), float(BF_low)

# Function to compute the 80% probability ellipse bounds
def find_80_percent_ellipse_values(data):
    mean_vector = data.mean()
    cov_matrix = np.cov(data.T)
    chi2_value = stats.chi2.ppf(0.80, df=2)  # 80% confidence level

    result_dict = {}
    for column in data.columns:
        mean = mean_vector[column]
        std_dev = np.sqrt(cov_matrix[data.columns.get_loc(column), data.columns.get_loc(column)])
        
        # Compute the lower and upper bounds
        lower_bound = mean - std_dev * np.sqrt(chi2_value)
        upper_bound = mean + std_dev * np.sqrt(chi2_value)
        
        result_dict[column] = {'mean': mean, 'lower_bound': lower_bound, 'upper_bound': upper_bound}
    
    return pd.DataFrame(result_dict).T


def create_figure_with_subplots(dataset, rd, variable_name, data_source,
                                title, obs_dict_ts,
                                storylines_dict_high_high_long,
                                storylines_dict_high_low_long,
                                time, time_obs, time_sl, time_sl_long):

    # ---- 2 rows instead of 1 ----
    fig, axs = plt.subplots(2, 3, figsize=(20, 10), dpi=300)

    BF_results = []

    # -----------------------
    # TOP ROW: time series
    # -----------------------
    for i in range(3):
        bf_high, bf_low = plot_mean_with_shading(
            axs[0, i],
            dataset[rd[i]],
            variable_name[i],
            data_source[i],
            title[i],
            obs_dict_ts[rd[i]],
            storylines_dict_high_high_long[rd[i]],
            storylines_dict_high_low_long[rd[i]],
            ['a)', 'b)', 'c)'][i],
            time, time_obs, time_sl, time_sl_long
        )

        BF_results.append([bf_high, bf_low])

    # -----------------------
    # BOTTOM ROW: Bayes factors
    # -----------------------
    for i in range(3):
        plot_bayes_panel(
            axs[1, i],
            BF_results[i],
            title=f"{['d)', 'e)', 'f)'][i]}"
        )

    # -----------------------
    # Final styling
    # -----------------------
    for ax in axs.flat:
        ax.tick_params(axis='both')

    plt.subplots_adjust(hspace=0.35)
    plt.tight_layout()

    return fig


def main(config):
    """Run the diagnostic."""
    #Reanalysis data
    ### Import ERA5 data
    ua_era5 = xr.open_dataset('/home/jmindlin/causal_EDJ/ERA5/ua_ERA5.nc')
    ua_era5 = ua_era5.rename({'latitude':'lat','longitude':'lon'})
    ta_era5 = xr.open_dataset('/home/jmindlin/causal_EDJ/ERA5/ta_ERA5.nc')
    ta_era5 = ta_era5.rename({'latitude':'lat','longitude':'lon'})

    del ua_era5

    ### Import JRA55 data

    ua_jra55_50 = [xr.open_dataset('/home/jmindlin/causal_EDJ/JRA55/ua/anl_mdl.033_ugrd.reg_tl319.'+str(year)+'01_'+str(year)+'12.mindlin756630_50hPa.nc') for year in np.arange(1958,2024,1)]
    ua_jra55_50_concat = xr.concat(ua_jra55_50,dim='initial_time0_hours')
    ua_jra55_50_concat = ua_jra55_50_concat.rename({'initial_time0_hours':'time','g4_lat_2':'lat','g4_lon_3':'lon'})

    ua_jra55_850 = [xr.open_dataset('/home/jmindlin/causal_EDJ/JRA55/ua/anl_mdl.033_ugrd.reg_tl319.'+str(year)+'01_'+str(year)+'12.mindlin756630_847hPa.nc') for year in np.arange(1958,2024,1)]
    ua_jra55_850_concat = xr.concat(ua_jra55_850,dim='initial_time0_hours')
    ua_jra55_850_concat = ua_jra55_850_concat.rename({'initial_time0_hours':'time','g4_lat_2':'lat','g4_lon_3':'lon'})

    ta_jra55 = [xr.open_dataset('/home/jmindlin/causal_EDJ/JRA55/ta/anl_mdl.011_tmp.reg_tl319.'+str(year)+'01_'+str(year)+'12.mindlin754486.nc') for year in np.arange(1958,2024,1)]
    ta_jra55_concat = xr.concat(ta_jra55,dim='initial_time0_hours')
    ta_jra55_concat = ta_jra55_concat.rename({'initial_time0_hours':'time','g4_lat_2':'lat','g4_lon_3':'lon','lv_HYBL1':'lev'})

    import urllib.request

    # URL of the data file
    url = "https://crudata.uea.ac.uk/cru/data/temperature/HadCRUT5.0Analysis_gl.txt"

    # Fetch the data from the URL
    with urllib.request.urlopen(url) as response:
        lines = response.read().decode('utf-8').splitlines()

    # Parse the lines to extract the data
    data = []
    months = []
    years = []
    for line in lines[::2]:
        values = line.split(' ')[2:-1]
        years.append(line.split(' ')[1])
        for i, value in enumerate(values):
            if value != '':
                data.append(value)
                months.append(i)

    # Convert the list of lists into a NumPy array
    data_array = np.array(data, dtype=float)
    data_array = data_array[:-12]

    # Print the resulting NumPy array
    print(data_array)

    time = pd.date_range(start='1850-01-01', end='2024-12-01', freq='MS')
    temperature_data = xr.DataArray(
        data_array, 
        coords={'time': time}, 
        dims='time', 
        name='temperature - HadCRU5'
    )

    tas_DJF = seasonal_data_months(temperature_data,[12,1,2])
    tas_DJF_anom = (tas_DJF - np.mean(tas_DJF.sel(time=slice('1950','1979')))).sel(time=slice('1950','2023'))

    tropical_warming = []
    tw_era5_DJF = seasonal_data_months(ta_era5,[12,1,2]).sel(lat=slice(15,-15)).mean(dim='lat').mean(dim='lon').sel(time=slice('1950','2023'))
    tw_era5_1950_2023_DJF = tw_era5_DJF - tw_era5_DJF.sel(time=slice('1950','1979')).mean(dim='time')
    tropical_warming.append(tw_era5_1950_2023_DJF.t)

    ### SST data
    sst_ERSST = xr.open_dataset('/home/jmindlin/causal_EDJ/SST_data/sst.mnmean_ERSST_2022_KAPLAN_grid.nc') #- xr.open_dataset('/home/jmindlin/causal_EDJ/SST_data/sst.mnmean_ERSST_2022_KAPLAN_grid.nc').mean(dim='lon')
    sst_ERSST_CP = sst_ERSST.sel(lon=slice(180,250)).sst.sel(lat=slice(-5,5)).mean(dim='lat').mean(dim='lon') 
    sst_ERSST_CP_DJF = seasonal_data_months(sst_ERSST_CP,[12,1,2])
    sst_ERSST_CP_DJF = sst_ERSST_CP_DJF - sst_ERSST_CP_DJF.sel(time=slice('1950','1979')).mean(dim='time')

    sst_ERSST_EP = sst_ERSST.sel(lon=slice(260,280)).sst.sel(lat=slice(0,10)).mean(dim='lat').mean(dim='lon')
    sst_ERSST_EP_DJF = seasonal_data_months(sst_ERSST_EP,[12,1,2])
    sst_ERSST_EP_DJF = sst_ERSST_EP_DJF - sst_ERSST_EP_DJF.sel(time=slice('1950','1979')).mean(dim='time')

    obs_dict_ts = {'gw':tas_DJF_anom.sel(time=slice('1950','2023')),'ta':tropical_warming[0].sel(time=slice('1950','2023')),
                   'tos_cp':sst_ERSST_CP_DJF.sel(time=slice('1950','2023'))}


    cfg=get_cfg(os.path.join(config["run_dir"],"settings.yml"))
    #print(cfg)
    meta_dataset = group_metadata(config["input_data"].values(), "dataset")
    models = []
    rd_list_models = []
    regressors_members = {}
    for dataset, dataset_list in meta_dataset.items(): ####DATASET es el modelo
        meta = group_metadata(config["input_data"].values(), "alias")
        if dataset != 'E3SM-1-0':
            print(f"Evaluate for {dataset}\n")
            models.append(dataset)
            rd_list_members = []
            for alias, alias_list in meta.items(): ###ALIAS son los miembros del ensemble para el modelo DATASET
                ts_dict = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice('1950','2099')) -  xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice('1940','1969')).mean(dim='time') for m in alias_list if (m["dataset"] == dataset) & (m["variable_group"] != 'ua850') & (m["variable_group"] != 'sst') & (m["variable_group"] != 'pr') & (m["variable_group"] != 'tos_zm') }
                if ('gw' in ts_dict.keys()) & (dataset == 'ACCESS-CM2'):
                    rd_list_members.append(ts_dict)
                    time = ts_dict['gw'].sel(time=slice('1950','2099')).time ### Model ensemble Means.time
                    time_obs = ts_dict['gw'].sel(time=slice('1950','2023')).time
                    time_sl = ts_dict['gw'].sel(time=slice('1950','2023')).time
                    time_sl_long = ts_dict['gw'].sel(time=slice('1950','2099')).time
                elif('gw' in ts_dict.keys()):
                    rd_list_members.append(ts_dict)
                else:
                    a = 'nada'

            #Index create data array
            regressor_names = rd_list_members[0].keys()
            regressors_members[dataset] = {}
            for rd in regressor_names:
                list_values = [rd_list_members[m][rd] for m,ensemble in enumerate(rd_list_members)]
                regressors_members[dataset][rd] = xr.concat(list_values, dim='ensemble') # Ensemble for each model 
                regressors_members[dataset][rd]['time'] = time

    regressor_names = rd_list_members[0].keys()
    regressors_members_MEM = {rd: xr.concat([regressors_members[ensemble_mean][rd].mean(dim='ensemble').sel(time=slice('1950','2099'))  for ensemble_mean in models], dim='model') for rd in regressor_names} ### Model ensemble Means
    regressors_members_MMEM = {rd: regressors_members_MEM[rd].mean(dim='model').sel(time=slice('1950','2099')) for rd in regressor_names} ### Model ensemble Means
    print(regressors_members_MEM)
    
    for rd in obs_dict_ts.keys():
        if rd == 'ua50_spv':
            obs_dict_ts[rd]['time'] = time_obs[:-2]
        else:
            obs_dict_ts[rd]['time'] = time_obs

    drivers = pd.read_csv(config["work_dir"]+'/remote_drivers/raw_remote_drivers_tropical_warming_global_warming_scaledGW.csv', index_col=0)
    sl_values = find_80_percent_ellipse_values(drivers)
    mean_GW = 1
    high_GW = 1.2
    low_GW = 0.8

    storylines_dict_high_high_long = {'gw':regressors_members_MMEM['gw'].sel(time=slice('1950','2099'))*high_GW,'ta':regressors_members_MMEM['gw'].sel(time=slice('1950','2099'))*mean_GW*sl_values['upper_bound']['ta'],
                   'tos_cp':regressors_members_MMEM['gw'].sel(time=slice('1950','2099'))*mean_GW*sl_values['upper_bound']['tos_cp']}

    storylines_dict_high_low_long = {'gw':regressors_members_MMEM['gw'].sel(time=slice('1950','2099'))*low_GW,'ta':regressors_members_MMEM['gw'].sel(time=slice('1950','2099'))*mean_GW*sl_values['lower_bound']['ta'],
                   'tos_cp':regressors_members_MMEM['gw'].sel(time=slice('1950','2099'))*mean_GW*sl_values['lower_bound']['tos_cp']}

    regressors_members_MEM_woGW = {rd: xr.concat([regressors_members[ensemble_mean][rd].mean(dim='ensemble').sel(time=slice('1950','2099'))  for ensemble_mean in models], dim='model') for rd in storylines_dict_high_low_long.keys()} ### Model ensemble Means

    fig = create_figure_with_subplots(regressors_members_MEM_woGW,list(storylines_dict_high_low_long.keys()),
                                      ['Global Warming [K]','Tropical Warming [K]','Central Pacific Warming [K]'],
                                      ['HadISSTv5','ERA5','ERSSTv5'],
                                      ['GW','TW/GW','CP/GW'],obs_dict_ts, storylines_dict_high_high_long, storylines_dict_high_low_long,
                                      time=time,time_obs=time_sl_long.sel(time=slice('1950','2023')),time_sl=time_sl,time_sl_long=time_sl_long)

    os.chdir(config["plot_dir"])
    os.getcwd()
    os.makedirs("remote_drivers",exist_ok=True)
    fig.savefig('/home/jmindlin/BF_codes/example_codes/Example6/dynamical_storylines_BFs_March2026_AAAAAAA.pdf',bbox_inches='tight')


# /climca/people/jmindlin/esmvaltool_output/full_storyline_analysis_complete_20240902_145448/run/multiple_regression_indices/multiple_regresion/settings.yml
if __name__ == "__main__":
    with run_diagnostic() as config:
        main(config)
                         

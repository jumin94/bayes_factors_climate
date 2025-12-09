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
    
def plot_mean_with_shading(ax, data, variable_name, data_source, title, obs, sl_high_high_l, sl_high_low_l, subplot_title, time, time_obs, sl_time, sl_time_long):
    """
    Plots the mean value across the 'ensemble' dimension with shading between
    the highest and lowest values for each time step on a given axis.
    
    Parameters:
    - ax: matplotlib.axes.Axes
        The axis to plot on.
    - data: xarray.Dataset
        The dataset containing the variable to plot.
    - variable_name: str
        The name of the variable to plot.
    - title: str
        The title of the subplot.
    """
    # Compute the mean, max, and min values across the 'ensemble' dimension
    mean_values = data.mean(dim='model')
    max_values = data.max(dim='model')
    min_values = data.min(dim='model')
    
    # Extract time and values for plotting
    mean_data = mean_values.values
    max_data = max_values.values
    min_data = min_values.values

    BF_rd_high = bayes_factor_RD(obs.sel(time=slice('1960', '2022')),
                                 make_xarr(sl_high_high_l.values, mean_values.sel(time=slice('1950', '2099')).time).sel(time=slice('1960', '2022')),
                                 mean_values.sel(time=slice('1960', '2022')))
    BF_rd_low = bayes_factor_RD(obs.sel(time=slice('1960', '2022')),
                                make_xarr(sl_high_low_l.values, mean_values.sel(time=slice('1950', '2099')).time).sel(time=slice('1960', '2022')),
                                mean_values.sel(time=slice('1960', '2022')))


    ax.set_title(subplot_title, fontsize=14, loc='left')

    # Plot model mean and spread
    if title == 'TW/GW':
        ax.plot(time, mean_data, label='CMIP6 hist+SSP5-8.5 MEM', color='black')
        ax.fill_between(time, min_data, max_data, color='grey', alpha=0.3, label='CMIP6 Spread')
        ax.plot(sl_time_long, sl_high_high_l.values, color='red', label=title + ', high storyline')
        ax.plot(sl_time_long, sl_high_low_l.values, color='blue', label=title + ', low storyline')
    else:
        ax.plot(time, mean_data, color='black')
        ax.fill_between(time, min_data, max_data, color='grey', alpha=0.3)
        ax.plot(sl_time_long, sl_high_high_l.values, color='red', label=title + ', high storyline')
        ax.plot(sl_time_long, sl_high_low_l.values, color='blue', label=title + ', low storyline')

    # Set labels and legend
    ax.set_xlabel('Year', fontsize=18)
    ax.set_ylabel(variable_name, fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14, length=10, width=2)
    plt.tick_params(axis='both', which='minor', labelsize=12, length=5, width=1.5)
    

    # ---------- BFs ----------
    BF_YMIN = 1e-4
    BF_YMAX = 100.0

    bf_high = max(float(BF_rd_high) if np.isfinite(BF_rd_high) else BF_YMIN, BF_YMIN)
    bf_low  = max(float(BF_rd_low)  if np.isfinite(BF_rd_low)  else BF_YMIN, BF_YMIN)
    bf_values = [bf_high, bf_low]
    bf_colors = ["red", "blue"]
    bar_positions = [0, 1]

    # ---------- Crear inset ----------
    ax_inset = inset_axes(ax,
                        width="28%", height="28%",  # más grande ~30%
                        loc="upper left",
                        borderpad=0.8)
    # Mover todo el inset más a la derecha y arriba para que no se superponga con eje Y
    ax_inset.set_position([0.12, 0.60, 0.30, 0.30])  # [left, bottom, width, height]

    # ---------- Sombreado ----------
    ax_inset.axhspan(0.1, 10, xmin=0.05, xmax=0.95, facecolor='gray', alpha=0.15, zorder=0)

    # ---------- Barras ----------
    bars = ax_inset.bar(bar_positions, bf_values, color=bf_colors, width=0.6,
                        edgecolor='k', linewidth=0.4, zorder=2)

    # ---------- Escala log y límites ----------
    ax_inset.set_yscale('log')
    ax_inset.set_ylim(BF_YMIN, BF_YMAX)

    # ---------- Ticks Y ----------
    for axis in ['both']:
        ax_inset.tick_params(axis=axis, which='major', labelsize=8, length=3)
        ax_inset.tick_params(axis=axis, which='minor', labelsize=8, length=2)

    ax_inset.yaxis.set_major_locator(LogLocator(base=10.0))
    ax_inset.yaxis.set_major_formatter(LogFormatterSciNotation(base=10))

    # ---------- Etiquetas eje X ----------
    x_labels = [f"high {title}", f"low {title}"]
    ax_inset.set_xticks(bar_positions)
    ax_inset.set_xticklabels(x_labels, fontsize=8, ha='center')

    # ---------- Texto encima de las barras ----------
    for xi, val in zip(bar_positions, bf_values):
        ax_inset.text(xi, val * 1.12, f"{val:.2g}", ha='center', va='bottom', fontsize=8, zorder=3)

    # ---------- Línea de referencia ----------
    ax_inset.axhline(1.0, color='gray', linestyle='--', linewidth=0.6, zorder=1)

    # ---------- Bordes del inset ----------
    for spine in ax_inset.spines.values():
        spine.set_linewidth(0.6)
        spine.set_visible(True)

    # ---------- Ajuste final del eje interno ----------
    # agregamos un poco de margen izquierdo para que las barras no toquen el eje Y
    ax_inset.margins(x=0.15)  # aumenta el espacio horizontal a los lados


    # optional: add tiny numeric labels above each bar (rounded)
    for xi, val in zip(bar_positions, bf_values):
        ax_inset.text(xi, val * 1.12, f"{val:.2g}", ha='center', va='bottom', fontsize=8)

def plot_mean_with_shading(ax, data, variable_name, data_source, title, obs, sl_high_high_l, sl_high_low_l, subplot_title, time, time_obs, sl_time, sl_time_long):
    """
    Plots the mean value across the 'ensemble' dimension with shading between
    the highest and lowest values for each time step on a given axis.
    Adds an inset bar plot for Bayes factors with better centering and custom ticks.
    """
    FS_SUBPLOT = 14  # uniform font size for titles and labels

    # Compute mean, min, max for ensemble
    mean_values = data.mean(dim='model')
    max_values = data.max(dim='model')
    min_values = data.min(dim='model')
    mean_data = mean_values.values
    max_data = max_values.values
    min_data = min_values.values

    # Plot observed data
    ax.plot(time_obs, obs.values, label=data_source, color='green', linewidth=2)

    # Plot model mean and spread
    ax.plot(time, mean_data, color='black')
    ax.fill_between(time, min_data, max_data, color='grey', alpha=0.3)
    ax.plot(sl_time_long, sl_high_high_l.values, color='red', label=title + ', high storyline')
    ax.plot(sl_time_long, sl_high_low_l.values, color='blue', label=title + ', low storyline')

    # Set axis labels and title
    ax.set_xlabel('Year', fontsize=FS_SUBPLOT)
    ax.set_ylabel(variable_name, fontsize=FS_SUBPLOT)
    ax.set_title(subplot_title, fontsize=FS_SUBPLOT, loc='left')
    ax.tick_params(axis='both', which='major', labelsize=FS_SUBPLOT)


    BF_rd_high = bayes_factor_RD(obs.sel(time=slice('1960', '2022')),
                                 make_xarr(sl_high_high_l.values, mean_values.sel(time=slice('1950', '2099')).time).sel(time=slice('1960', '2022')),
                                 mean_values.sel(time=slice('1960', '2022')))
    BF_rd_low = bayes_factor_RD(obs.sel(time=slice('1960', '2022')),
                                make_xarr(sl_high_low_l.values, mean_values.sel(time=slice('1950', '2099')).time).sel(time=slice('1960', '2022')),
                                mean_values.sel(time=slice('1960', '2022')))
    
    # --- Inset bar plot for Bayes factors ---
    FS_BAR = 16  # font size for inset bar plot (twice as large)
    BF_YMIN = 1e-4
    BF_YMAX = 100.0
    bf_high = max(float(BF_rd_high) if np.isfinite(BF_rd_high) else BF_YMIN, BF_YMIN)
    bf_low  = max(float(BF_rd_low)  if np.isfinite(BF_rd_low)  else BF_YMIN, BF_YMIN)
    bf_values = [bf_high, bf_low]
    bf_colors = ["red", "blue"]
    bar_positions = [0, 1]

    # Inset axes, closer to the left
    ax_inset = inset_axes(ax,
                          width="35%", height="35%",
                          loc="upper center",  # initial location
                          borderpad=0.8)
    
    # Manual position adjustment: ~0.2 from left
    ax_inset.set_position([0.20, 0.55, 0.35, 0.35])  # [left, bottom, width, height]

    # Shading behind bars
    ax_inset.axhspan(0.1, 10, xmin=0.05, xmax=0.95, facecolor='gray', alpha=0.15, zorder=0)

    # Bars
    ax_inset.bar(bar_positions, bf_values, color=bf_colors, width=0.6, edgecolor='k', linewidth=0.4, zorder=2)

    # Log scale
    ax_inset.set_yscale('log')
    ax_inset.set_ylim(BF_YMIN, BF_YMAX)
    ax_inset.axhline(1.0, color='gray', linestyle='--', linewidth=0.6, zorder=1)

    # Custom x-tick labels
    if title == 'GW':
        xticks_labels = ["low GW", "high GW"]
    elif title == 'TW/GW':
        xticks_labels = ["low \n TW/GW", "high \n TW/GW"]
    elif title == 'CP/GW':
        xticks_labels = ["low \n CP/GW", "high \n CP/GW"]
    else:
        xticks_labels = ["low", "high"]

    ax_inset.set_xticks(bar_positions)
    ax_inset.set_xticklabels(xticks_labels, fontsize=FS_BAR, ha='center')

    # Minor tick formatting
    ax_inset.yaxis.set_major_locator(LogLocator(base=10.0))
    ax_inset.yaxis.set_major_formatter(LogFormatterSciNotation(base=10))
    ax_inset.tick_params(axis='both', which='major', labelsize=FS_BAR, length=3)
    ax_inset.tick_params(axis='both', which='minor', labelsize=FS_BAR, length=2)

    # Values on top of bars
    for xi, val in zip(bar_positions, bf_values):
        ax_inset.text(xi, val * 1.12, f"{val:.2g}", ha='center', va='bottom', fontsize=FS_BAR)



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


def create_figure_with_subplots(dataset, rd, variable_name,data_source,title,obs_dict_ts,storylines_dict_high_high_long, storylines_dict_high_low_long,time,time_obs,time_sl,time_sl_long):
    """
    Creates a figure with four subplots, each plotting the mean value with shading between
    the highest and lowest values for each time step.
    
    Parameters:
    - dataset: xarray.Dataset
        The dataset containing the variable to plot.
    - variable_name: str
        The name of the variable to plot.
    """
    fig, axs = plt.subplots(1, 3, figsize=(20, 6),dpi=300)
    
    for i in range(3):
        ax = axs.flat[i]
        subplot_labels = ['a)','b)','c)']
        plot_mean_with_shading(ax, dataset[rd[i]], variable_name[i],data_source[i],title[i],obs_dict_ts[rd[i]],storylines_dict_high_high_long[rd[i]], storylines_dict_high_low_long[rd[i]], subplot_labels[i], time,time_obs,time_sl,time_sl_long)
    
    # Add labels and grid to each subplot
    for i, ax in enumerate(axs.flat):
        
        # Customize ticks
        ax.tick_params(axis='both', which='major', labelsize=15)
        
        if i != 5:
            # Add grid
            ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)

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
    fig.savefig('/home/jmindlin/BF_codes/example_codes/Example6/dynamical_storylines_BFs.pdf',bbox_inches='tight')


# /climca/people/jmindlin/esmvaltool_output/full_storyline_analysis_complete_20240902_145448/run/multiple_regression_indices/multiple_regresion/settings.yml
if __name__ == "__main__":
    with run_diagnostic() as config:
        main(config)
                         
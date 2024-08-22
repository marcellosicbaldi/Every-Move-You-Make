import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def detect_HR_change(bursts, hr_df, off, plot = False):
    """
    Detect HR change from RR intervals. RR intervals are already corrected using the kubios method.

    Parameters
    ----------
    bursts : pd.DataFrame
        DataFrame with burst start times, end times, and duration
    hr_df : pd.Series
        HR signal
    plot : bool, optional
        If True, plot the HR signal and the detected HR change (for debugging purposes)
    Returns
    -------
    bursts : pd.DataFrame
        Input DataFrame with added HR response, HR response normalized, HR baseline, HR peak, and HR peak latency 
    """

    bursts["HR_response"] = [None] * len(bursts)
    bursts["HR_response_normalized"] = [None] * len(bursts)
    for i in range(len(bursts)):
        hr_burst = hr_df.loc[bursts["Start"].iloc[i] - pd.Timedelta(seconds = 19+off):bursts["Start"].iloc[i] + pd.Timedelta(seconds = 50-off)]
        if hr_burst.isna().sum() > 0: # missing HR data
            continue
            
        df_burst = pd.DataFrame({'HR': hr_burst}, index = hr_burst.index)
        HR_baseline = df_burst.loc[bursts["Start"].iloc[i] - pd.Timedelta(seconds = 18+off):bursts["Start"].iloc[i] - pd.Timedelta(seconds = 8+off)]["HR"].mean()

        bursts.at[i, "HR_response"] = df_burst["HR"].values

        # express it as a percentage of the baseline
        df_burst["HR"] = df_burst["HR"] / HR_baseline * 100 - 100
        bursts.at[i, "HR_response_normalized"] = df_burst["HR"].values
        
        bursts.loc[i, "HR_baseline"] = HR_baseline
        bursts.loc[i, "HR_peak"] = df_burst["HR"].iloc[20:].max()

        if plot:
            plt.figure()
            plt.plot(hr_burst.index, hr_burst)
            plt.plot(df_burst['HR'])
            plt.axvline(x = bursts["Start"].iloc[i], color = 'r')
            plt.axhline(y = HR_baseline, color = 'r', linestyle = '--')
        df_burst.index = np.arange(-19, 50, 1)

        # latency of the HR peak
        bursts.loc[i, "HR_peak_latency"] = df_burst["HR"].iloc[20:].idxmax()
    return bursts

def detect_ACC_change(bursts, env_diff_dict, off):

    """
    Detect ACC change from envelope of the acceleration signals

    Parameters
    ----------
    bursts : pd.DataFrame
        DataFrame with burst start times, end times, and duration
    env_diff_dict : dict
        Dictionary with envelope of the acceleration signals. Keys are "T", "LL", "RL", "LW", "RW"

    Returns
    -------
    bursts : pd.DataFrame
        Input DataFrame with added ACC response
    """
    
    bursts["ACC_response"] = [None] * len(bursts)

    for i in range(len(bursts)):
        limb_burst_total = np.zeros(69)
        for limb in ["T", "LL", "RL", "LW", "RW"]:
            if limb in bursts["Limbs"].iloc[i]:
                limb_burst = env_diff_dict[limb].loc[bursts["Start"].iloc[i] - pd.Timedelta(seconds = 19+off):bursts["Start"].iloc[i] + pd.Timedelta(seconds = 50-off)]
                limb_baseline = limb_burst.loc[bursts["Start"].iloc[i] - pd.Timedelta(seconds = 18+off):bursts["Start"].iloc[i] - pd.Timedelta(seconds = 8+off)].mean()
                limb_burst = limb_burst / limb_baseline * 100 - 100
                limb_burst_total += limb_burst
        bursts.at[i, "ACC_response"] = limb_burst_total.tolist()

    return bursts

#### Coherent averaaging functions ####

def coherent_avg(data, method = "mean"):
    """
    Compute the coherent average of a list of data arrays

    Parameters
    ----------
    data : list
        List of data arrays

    Returns
    -------
    avg : np.array
        Coherent average
    """
    try:
        data_all = np.zeros((len(data), len(data[0]["HR"])))
    except:
        return np.nan, np.nan
    for i, dat in enumerate(data):
        data_all[i] = dat["HR"].values
    if method == "mean":
        HR_burst_avg_lw = np.nanmean(data_all, axis = 0)
    elif method == "median":
        HR_burst_avg_lw = np.nanmedian(data_all, axis = 0)
    HR_burst_std_lw = np.nanstd(data_all, axis = 0)

    return HR_burst_avg_lw, HR_burst_std_lw

def coherent_avg_ACC(data, method = "mean"):
    try:
        data_all = np.zeros((len(data), len(data[0]["ACC"])))
    except:
        return np.nan, np.nan
    for i, dat in enumerate(data):
        try:
            data_all[i] = dat["ACC"].values
        except:
            data_all[i] = dat["ACC"].values[:-1]
    if method == "mean":
        ACC_burst_avg_lw = np.nanmean(data_all, axis = 0)
    elif method == "median":
        ACC_burst_avg_lw = np.nanmedian(data_all, axis = 0)
    ACC_burst_std_lw = np.nanstd(data_all, axis = 0)

    return ACC_burst_avg_lw, ACC_burst_std_lw
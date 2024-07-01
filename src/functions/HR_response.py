import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def detect_HR_change_from_RR(bursts, hr_df, off, plot = False):
    """
    Detect HR change from RR intervals. RR intervals are already corrected using the kubios method.

    Parameters
    ----------
    bursts : pd.DataFrame
        DataFrame with burst start times, end times, and duration
    hr_df : pd.Series
        HR signal
    plot : bool, optional
        If True, plot the HR signal and the detected HR change  

    Returns
    -------
    HR_change_max : list
        List with the maximum HR change (HR peak) for each burst
    latency_HR_change_max : list
        List with the latency of the maximum HR change (HR peak) for each burst
    duration_HR_response : list
        List with the duration of the HR response for each burst (defined as the time between movement onset and the HR return to baseline)    
    """
    HR_bursts_df = []
    HR_change_max = []
    latency_HR_change_max = []
    duration_HR_response = []
    posture_change = []
    AUC = []
    for i in range(len(bursts)):
        hr_burst = hr_df.loc[bursts["Start"].iloc[i] - pd.Timedelta(seconds = 19+off):bursts["Start"].iloc[i] + pd.Timedelta(seconds = 40-off)]
        if hr_burst.isna().sum() > 0:
            # print(f"Missing HR data for burst {i}")
            continue
            
        df_burst = pd.DataFrame({'HR': hr_burst}, index = hr_burst.index)
        HR_baseline = df_burst.loc[bursts["Start"].iloc[i] - pd.Timedelta(seconds = 18+off):bursts["Start"].iloc[i] - pd.Timedelta(seconds = 8+off)]["HR"].mean()

        # baseline correction
        # df_burst["HR"] = df_burst["HR"] - HR_baseline
        # express it as a percentage of the baseline
        df_burst["HR"] = df_burst["HR"] / HR_baseline * 100 - 100
        # df_burst["t"] = np.arange(-19, 40, 1)
        HR_bursts_df.append(df_burst)
        HR_change_max.append(df_burst["HR"].iloc[20:].max())
        # latency = df_burst["HR"].idxmax() - bursts["Start"].iloc[i]
        latency_HR_change_max.append(df_burst["HR"].idxmax() - bursts["Start"].iloc[i]) # TODO start from .iloc[20:] but requires to adjust the index 
        # time from movement onset to HR return to baseline
        # duration_HR_response.append(   

        # if np.isnan(bursts["posture_change"].iloc[i]):
        #     posture_change.append(0)
        # else:
        #     posture_change.append(bursts["posture_change"].iloc[i])

        AUC.append(bursts["AUC"].iloc[i])

        if plot:
            plt.figure()
            plt.plot(hr_burst.index, hr_burst)
            plt.plot(df_burst['HR'])
            plt.axvline(x = bursts["Start"].iloc[i], color = 'r')
            plt.axhline(y = HR_baseline, color = 'r', linestyle = '--')

    return HR_bursts_df, np.array(HR_change_max), np.array(latency_HR_change_max), np.array(AUC), np.array(posture_change)

def detect_ACC_change(bursts, acc_df, off, plot = False):
    ACC_bursts_df = []
    AUC = []
    posture_change = []
    acc_df = acc_df.resample('1s').std()
    for i in range(len(bursts)):
        acc_burst = acc_df.loc[bursts["start"].iloc[i] - pd.Timedelta(seconds = 19+off):bursts["start"].iloc[i] + pd.Timedelta(seconds = 40-off)]

        df_burst = pd.DataFrame({'ACC': acc_burst}, index = acc_burst.index)
        ACC_baseline = df_burst.loc[bursts["start"].iloc[i] - pd.Timedelta(seconds = 18+off):bursts["start"].iloc[i] - pd.Timedelta(seconds = 8+off)]["ACC"].mean()

        # baseline correction
        df_burst["ACC"] = df_burst["ACC"] / ACC_baseline * 100 - 100
        # df_burst["t"] = np.arange(-19, 40, 1)
        ACC_bursts_df.append(df_burst)
        AUC.append(bursts["AUC"].iloc[i])
        if np.isnan(bursts["posture_change"].iloc[i]):
            posture_change.append(0)
        else:
            posture_change.append(bursts["posture_change"].iloc[i])
    return ACC_bursts_df, np.array(AUC), np.array(posture_change)

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
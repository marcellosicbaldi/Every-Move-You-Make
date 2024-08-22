import numpy as np
import pandas as pd
import pickle
import pyreadr
import neurokit2 as nk

from sleep_diary import diary_SPT, diary_TIB

from functions.bursts import characterize_bursts, filter_bursts, compute_envelope
from functions.HR_response import detect_HR_change, detect_ACC_change

############# Merge SIB information with bursts #############

part2_outputFolder = "/Volumes/Untitled/rehab/GGIR/GGIR_output_lw_TIB/output_lw_data/meta/ms2.out/"
part3_outputFolder = "/Volumes/Untitled/rehab/GGIR/GGIR_output_lw_TIB/output_lw_data/meta/ms3.out/"
subjects = ["158", "098", "633", "279", "906", "547", "971", "958", "815"]

SIB_GGIR = {sub: pyreadr.read_r(part3_outputFolder + "LW_" + sub + ".CWA.RData")['sib.cla.sum'][["sib.onset.time", "sib.end.time"]] for sub in subjects}

bursts_lw = {sub: 0 for sub in subjects}
bursts_rw = {sub: 0 for sub in subjects}
bursts_ll = {sub: 0 for sub in subjects}
bursts_rl = {sub: 0 for sub in subjects}
bursts_trunk = {sub: 0 for sub in subjects}
bursts_all_limbs = {sub: 0 for sub in subjects}
bursts_all_limbs_new = {sub: 0 for sub in subjects}
SIB = {sub: 0 for sub in subjects}

bursts_df = pd.DataFrame()

for i, sub in enumerate(subjects):
    SIB_GGIR[sub]["sib.onset.time"] = pd.to_datetime(SIB_GGIR[sub]["sib.onset.time"].values).tz_localize(None)
    SIB_GGIR[sub]["sib.end.time"] = pd.to_datetime(SIB_GGIR[sub]["sib.end.time"].values).tz_localize(None)
    SIB_GGIR[sub]["sib.duration"] = SIB_GGIR[sub]["sib.end.time"] - SIB_GGIR[sub]["sib.onset.time"]

    with open(f'/Volumes/Untitled/rehab/data/{sub}/bursts_FINAL_envInterp_p2p.pkl', 'rb') as f:
        bursts = pickle.load(f)

    df_merged_intervals = characterize_bursts(bursts)
    spt_start = diary_SPT[sub][0] - pd.Timedelta('10 min')
    spt_end = diary_TIB[sub][1] + pd.Timedelta('5 min')

    SIB[sub] = SIB_GGIR[sub][(SIB_GGIR[sub]["sib.onset.time"] >= spt_start) & (SIB_GGIR[sub]["sib.end.time"] <= spt_end)].reset_index(drop=True)
    SIB[sub] = SIB_GGIR[sub][(SIB_GGIR[sub]["sib.onset.time"] >= spt_start) & (SIB_GGIR[sub]["sib.end.time"] <= spt_end)].reset_index(drop=True)

    # Take df_merged_intervals between spt_start and spt_end
    df_merged_intervals = df_merged_intervals[(df_merged_intervals["Start"] >= spt_start) & (df_merged_intervals["End"] <= spt_end)].reset_index(drop=True) 

    SIB[sub]["awake.duration"] = SIB[sub]["sib.onset.time"].shift(-1) - SIB[sub]["sib.end.time"]
    SIB[sub]["sub_ID"] = sub

    df_merged_intervals["SIB"] = 0
    for i, row in SIB[sub].iterrows():
        df_merged_intervals.loc[(df_merged_intervals["Start"] >= row["sib.onset.time"] + pd.Timedelta("5s")) & (df_merged_intervals["End"] <= row["sib.end.time"] - pd.Timedelta("5s")), "SIB"] = 1

    df_merged_intervals["sub_ID"] = sub

    start_sleep = diary_SPT[sub][0]
    end_sleep = diary_SPT[sub][1]

    df_merged_intervals = df_merged_intervals.loc[(df_merged_intervals["Start"] >= start_sleep) & (df_merged_intervals["End"] <= end_sleep)]

    bursts_df = pd.concat([bursts_df, df_merged_intervals])

bursts_df.reset_index(drop=True, inplace=True)

offsets = {"158": 4, "098": 4, "633": 4, "279": 5, "906": 3, "547": 5, "971": 3, "958": 4, "815": 4} # 971 and 906 spostare di 2 e di 1 a destra

bursts_HR = pd.DataFrame()

for i, sub in enumerate(subjects):

    print(sub)

    off = offsets[sub]

    start_sleep, end_sleep = diary_SPT[sub]
    
    ## ECG processing ##
    ecg_df = pd.read_pickle(f'/Volumes/Untitled/rehab/data/{sub}/polar_processed/ecg.pkl')

    ecg_df = ecg_df.loc[start_sleep:end_sleep]
    ecg_filtered = nk.ecg_clean(ecg_df.values, sampling_rate=130)

    # Extract peaks
    _, results = nk.ecg_peaks(ecg_filtered, sampling_rate=130, method = 'neurokit')
    rpeaks = results["ECG_R_Peaks"]
    _, rpeaks_corrected = nk.signal_fixpeaks(rpeaks, sampling_rate=130, iterative=True, method="Kubios")

    t_rpeaks = ecg_df.index.to_series().values[rpeaks]
    t_rpeaks_corrected = ecg_df.index.to_series().values[rpeaks_corrected]
    rr = np.diff(t_rpeaks).astype('timedelta64[ns]').astype('float64') / 1000000000
    rr_corrected = np.diff(t_rpeaks_corrected).astype('timedelta64[ns]').astype('float64') / 1000000000
    hr_ecg = 60/rr
    hr_ecg_corrected = 60/rr_corrected
    hr_df = pd.Series(hr_ecg_corrected, index = t_rpeaks_corrected[1:]).resample("1 s").mean()#.rolling('10s', min_periods=1, center=True).mean()
    hr_df = hr_df.interpolate(method = 'cubic')
    hr_df_noncorrected = pd.Series(hr_ecg, index = t_rpeaks[1:]).resample("1 s").mean()
    hr_df_noncorrected = hr_df_noncorrected.interpolate(method = 'linear')

    artifacts_ecg = pd.read_csv(f'/Volumes/Untitled/rehab/data/{sub}/polar_processed/artifacts_ecg.csv')
    artifacts_ecg['Start'] = pd.to_datetime(artifacts_ecg['Start']).apply(lambda x: x.replace(tzinfo=None))
    artifacts_ecg['End'] = pd.to_datetime(artifacts_ecg['End']).apply(lambda x: x.replace(tzinfo=None))

    for i in range(len(artifacts_ecg)):
        hr_df.loc[artifacts_ecg["Start"].iloc[i]:artifacts_ecg["End"].iloc[i]] = np.nan

    hr_df.interpolate(method = 'cubic', inplace = True)

    for i in range(len(artifacts_ecg)):
        if artifacts_ecg["End"].iloc[i] - artifacts_ecg["Start"].iloc[i] > pd.Timedelta("20 s"):
            hr_df.loc[artifacts_ecg["Start"].iloc[i]:artifacts_ecg["End"].iloc[i]] = np.nan

    ## ACC processing ##

    trunk_df = pd.read_pickle(f"/Volumes/Untitled/rehab/data/{sub}/acc_night/trunk.pkl") * 1000
    ll_df = pd.read_pickle(f"/Volumes/Untitled/rehab/data/{sub}/acc_night/ll.pkl") * 1000
    rl_df = pd.read_pickle(f"/Volumes/Untitled/rehab/data/{sub}/acc_night/rl.pkl") * 1000
    lw_df = pd.read_pickle(f"/Volumes/Untitled/rehab/data/{sub}/acc_night/lw.pkl") * 1000
    rw_df = pd.read_pickle(f"/Volumes/Untitled/rehab/data/{sub}/acc_night/rw.pkl") * 1000

    env_diff_T = compute_envelope(trunk_df, resample = True).resample("1 s").mean()
    env_diff_LL = compute_envelope(ll_df, resample = True).resample("1 s").mean()
    env_diff_RL = compute_envelope(rl_df, resample = True).resample("1 s").mean()
    env_diff_LW = compute_envelope(lw_df, resample = True).resample("1 s").mean()
    env_diff_RW = compute_envelope(rw_df, resample = True).resample("1 s").mean()

    env_diff_dict = {"T": env_diff_T, "LL": env_diff_LL, "RL": env_diff_RL, "LW": env_diff_LW, "RW": env_diff_RW}

    ## Detect HR response ##

    bursts_sub = filter_bursts(bursts_df.loc[bursts_df["sub_ID"] == sub].reset_index(drop=True)).reset_index(drop=True)
    
    bursts_ACC = detect_ACC_change(bursts_sub, env_diff_dict, offsets[sub]-2) # -2 OK

    bursts_HR_sub = detect_HR_change(bursts_ACC, hr_df, offsets[sub]-1, plot = False)

    bursts_HR = pd.concat([bursts_HR, bursts_HR_sub])

bursts_HR.to_pickle("/Volumes/Untitled/rehab/data/bursts_HR_ACC_final.pkl")


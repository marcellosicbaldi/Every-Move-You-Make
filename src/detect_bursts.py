# Description: This script detects bursts in the accelerometer data of the trunk and limbs of the subjects, and save the results in a pickle file. 
# The bursts are detected using the Hilbert envelope method, and the isolated movements are extracted for each limb. 
# The script also computes the area under the curve of the Hilbert envelope for each burst, and detects posture changes from the trunk accelerometer data. 
# The results are saved in a dictionary with keys for each combination of limbs.

import numpy as np
import pandas as pd
import neurokit2 as nk
import pickle

from functions.bursts import hl_envelopes_idx, detect_bursts, merge_excluding, is_isolated, find_isolated_combination, find_combined_movements_all_limbs
from functions.acc_utils import compute_acc_norm
from functions.posture import compute_spherical_coordinates, detect_posture_changes


diary_SPT = {    
    "158": [pd.Timestamp('2024-02-28 23:00:00'), pd.Timestamp('2024-02-29 07:15:00')], # 158 OK
    "633": [pd.Timestamp('2024-03-07 00:05:00'), pd.Timestamp('2024-03-07 06:36:00')], # 633 OK
    "906": [pd.Timestamp('2024-03-07 00:30:00'), pd.Timestamp('2024-03-07 07:30:00')], # 906 OK
    "958": [pd.Timestamp('2024-03-13 22:00:00'), pd.Timestamp('2024-03-14 06:00:00')], # 958 OK
    "127": [pd.Timestamp('2024-03-13 23:15:00'), pd.Timestamp('2024-03-14 06:50:00')], # 127 OK
    "098": [pd.Timestamp('2024-03-16 02:01:00'), pd.Timestamp('2024-03-16 09:50:00')], # 098 OK
    "547": [pd.Timestamp('2024-03-16 01:04:00'), pd.Timestamp('2024-03-16 07:40:00')], # 547 OK
    "815": [pd.Timestamp('2024-03-20 23:00:00'), pd.Timestamp('2024-03-21 07:30:00')], # 815 OK
    "914": [pd.Timestamp('2024-03-20 21:50:00'), pd.Timestamp('2024-03-21 05:50:00')], # 914 OK
    "971": [pd.Timestamp('2024-03-20 23:50:00'), pd.Timestamp('2024-03-21 07:50:00')], # 971 OK
    "279": [pd.Timestamp('2024-03-28 00:10:00'), pd.Timestamp('2024-03-28 07:27:00')], # 279 OK
    "965": [pd.Timestamp('2024-03-28 01:25:00'), pd.Timestamp('2024-03-28 09:20:00')], # 965 OK
}

diary_TIB = {
    "158": [pd.Timestamp('2024-02-28 22:15:00'), pd.Timestamp('2024-02-29 07:45:00')], # 158 OK
    "633": [pd.Timestamp('2024-03-06 23:39:00'), pd.Timestamp('2024-03-07 08:00:00')], # 633 OK 
    "906": [pd.Timestamp('2024-03-07 00:15:00'), pd.Timestamp('2024-03-07 07:35:00')], # 906 OK
    "958": [pd.Timestamp('2024-03-13 21:30:00'), pd.Timestamp('2024-03-14 06:30:00')], # 958 OK
    "127": [pd.Timestamp('2024-03-13 22:00:00'), pd.Timestamp('2024-03-14 07:10:00')], # 127 OK 
    "098": [pd.Timestamp('2024-03-16 01:49:00'), pd.Timestamp('2024-03-16 09:52:00')], # 098 OK 
    "547": [pd.Timestamp('2024-03-16 00:26:00'), pd.Timestamp('2024-03-16 08:20:00')], # 547 OK 
    "815": [pd.Timestamp('2024-03-20 22:00:00'), pd.Timestamp('2024-03-21 07:30:00')], # 815 OK 
    "914": [pd.Timestamp('2024-03-20 21:30:00'), pd.Timestamp('2024-03-21 06:20:00')], # 914 OK 
    "971": [pd.Timestamp('2024-03-20 23:30:00'), pd.Timestamp('2024-03-21 08:08:00')], # 971 OK 
    "279": [pd.Timestamp('2024-03-28 00:04:00'), pd.Timestamp('2024-03-28 07:41:00')], # 279 OK
    "965": [pd.Timestamp('2024-03-28 01:22:00'), pd.Timestamp('2024-03-28 09:22:00')], # 965 OK
}

subjects = ["158", "098", "633", "279", "906", "547", "971", "958", "815"]

for i, sub in enumerate(subjects):

    print(sub)

    with open(f'/Volumes/Untitled/rehab/data/{sub}/ax_data.pkl', 'rb') as f:
        ax_data = pickle.load(f)

    print("Loaded ax_data!")

    trunk_df = pd.Series(compute_acc_norm(ax_data["trunk"][["x", "y", "z"]].values), index = pd.to_datetime(ax_data["trunk"]["time"], unit = "s") + pd.Timedelta(hours = 1))
    ll_df = pd.Series(compute_acc_norm(ax_data["la"][["x", "y", "z"]].values), index = pd.to_datetime(ax_data["la"]["time"], unit = "s") + pd.Timedelta(hours = 1))
    rl_df = pd.Series(compute_acc_norm(ax_data["ra"][["x", "y", "z"]].values), index = pd.to_datetime(ax_data["ra"]["time"], unit = "s") + pd.Timedelta(hours = 1))
    lw_df = pd.Series(compute_acc_norm(ax_data["lw"][["x", "y", "z"]].values), index = pd.to_datetime(ax_data["lw"]["time"], unit = "s") + pd.Timedelta(hours = 1))
    rw_df = pd.Series(compute_acc_norm(ax_data["rw"][["x", "y", "z"]].values), index = pd.to_datetime(ax_data["rw"]["time"], unit = "s") + pd.Timedelta(hours = 1))

    start_sleep, end_sleep = diary_TIB[sub]

    trunk_df = trunk_df.loc[start_sleep:end_sleep]
    ll_df = ll_df.loc[start_sleep:end_sleep]
    rl_df = rl_df.loc[start_sleep:end_sleep]
    lw_df = lw_df.loc[start_sleep:end_sleep]
    rw_df = rw_df.loc[start_sleep:end_sleep]

    # TODO: Modify sampling rate to 100 Hz

    lw_df_bp = pd.Series(nk.signal_filter(lw_df.values, sampling_rate = 50, lowcut=0.1, highcut=5, method='butterworth', order=8), index = lw_df.index)
    rw_df_bp = pd.Series(nk.signal_filter(rw_df.values, sampling_rate = 50, lowcut=0.1, highcut=5, method='butterworth', order=8), index = rw_df.index)
    ll_df_bp = pd.Series(nk.signal_filter(ll_df.values, sampling_rate = 50, lowcut=0.1, highcut=5, method='butterworth', order=8), index = ll_df.index)
    rl_df_bp = pd.Series(nk.signal_filter(rl_df.values, sampling_rate = 50, lowcut=0.1, highcut=5, method='butterworth', order=8), index = rl_df.index)
    trunk_df_bp = pd.Series(nk.signal_filter(trunk_df.values, sampling_rate = 50, lowcut=0.1, highcut=5, method='butterworth', order=8), index = trunk_df.index)
    bursts_lw = detect_bursts(lw_df_bp, plot = False, alfa = 7)
    bursts_rw = detect_bursts(rw_df_bp, plot = False, alfa = 7)
    bursts_ll = detect_bursts(ll_df_bp, plot = False, alfa = 6)
    bursts_rl = detect_bursts(rl_df_bp, plot = False, alfa = 6)
    bursts_trunk = detect_bursts(trunk_df_bp, plot = False, alfa = 5)

    # Isolation checks
    bursts_ll['isolated'] = bursts_ll.apply(lambda x: is_isolated(x['start'], x['end'], merge_excluding(bursts_ll)), axis=1)
    bursts_rl['isolated'] = bursts_rl.apply(lambda x: is_isolated(x['start'], x['end'], merge_excluding(bursts_rl)), axis=1)
    bursts_lw['isolated'] = bursts_lw.apply(lambda x: is_isolated(x['start'], x['end'], merge_excluding(bursts_lw)), axis=1)
    bursts_rw['isolated'] = bursts_rw.apply(lambda x: is_isolated(x['start'], x['end'], merge_excluding(bursts_rw)), axis=1)
    bursts_trunk['isolated'] = bursts_trunk.apply(lambda x: is_isolated(x['start'], x['end'], merge_excluding(bursts_trunk)), axis=1)

    # Extract isolated movements for each limb
    bursts_ll_isolated = bursts_ll[bursts_ll['isolated']]
    bursts_rl_isolated = bursts_rl[bursts_rl['isolated']]
    bursts_lw_isolated = bursts_lw[bursts_lw['isolated']]
    bursts_rw_isolated = bursts_rw[bursts_rw['isolated']]
    bursts_trunk_isolated = bursts_trunk[bursts_trunk['isolated']]

    bursts_wrists_isolated = pd.concat([bursts_lw_isolated, bursts_rw_isolated], ignore_index=True)
    bursts_legs_isolated = pd.concat([bursts_ll_isolated, bursts_rl_isolated], ignore_index=True)

    bursts_both_wrists = find_isolated_combination([bursts_lw, bursts_rw], [bursts_ll, bursts_rl, bursts_trunk]).iloc[::2].reset_index(drop=True)

    # Finding isolated movements for both legs alone (no wrists or trunk)
    bursts_both_legs = find_isolated_combination([bursts_ll, bursts_rl], [bursts_lw, bursts_rw, bursts_trunk]).iloc[::2].reset_index(drop=True)

    bursts_lw["limb"] = "lw"
    bursts_rw["limb"] = "rw"
    bursts_ll["limb"] = "ll"
    bursts_rl["limb"] = "rl"
    bursts_trunk["limb"] = "trunk"
    bursts_all_limbs_combined = find_combined_movements_all_limbs([bursts_lw, bursts_rw, bursts_ll, bursts_rl, bursts_trunk])

    bursts_all_limbs_combined["AUC"] = np.nan

    lmin, lmax = hl_envelopes_idx(lw_df_bp.values, dmin=9, dmax=9)
    if len(lmin) > len(lmax):
        lmin = lmin[:-1]
    if len(lmax) > len(lmin):
        lmax = lmax[1:]
    env_diff_lw = pd.Series(lw_df_bp.values[lmax] - lw_df_bp.values[lmin], index = lw_df_bp.index[lmax])

    lmin, lmax = hl_envelopes_idx(rw_df_bp.values, dmin=9, dmax=9)
    if len(lmin) > len(lmax):
        lmin = lmin[:-1]
    if len(lmax) > len(lmin):
        lmax = lmax[1:]
    env_diff_rw = pd.Series(rw_df_bp.values[lmax] - rw_df_bp.values[lmin], index = rw_df_bp.index[lmax])

    lmin, lmax = hl_envelopes_idx(ll_df_bp.values, dmin=9, dmax=9)
    if len(lmin) > len(lmax):
        lmin = lmin[:-1]
    if len(lmax) > len(lmin):
        lmax = lmax[1:]
    env_diff_ll = pd.Series(ll_df_bp.values[lmax] - ll_df_bp.values[lmin], index = ll_df_bp.index[lmax])

    lmin, lmax = hl_envelopes_idx(rl_df_bp.values, dmin=9, dmax=9)
    if len(lmin) > len(lmax):
        lmin = lmin[:-1]
    if len(lmax) > len(lmin):
        lmax = lmax[1:]
    env_diff_rl = pd.Series(rl_df_bp.values[lmax] - rl_df_bp.values[lmin], index = rl_df_bp.index[lmax])

    lmin, lmax = hl_envelopes_idx(trunk_df_bp.values, dmin=9, dmax=9)
    if len(lmin) > len(lmax):
        lmin = lmin[:-1]
    if len(lmax) > len(lmin):
        lmax = lmax[1:]
    env_diff_trunk = pd.Series(trunk_df_bp.values[lmax] - trunk_df_bp.values[lmin], index = trunk_df_bp.index[lmax])

    for i, b in enumerate(range(len(bursts_all_limbs_combined))):
        bursts_all_limbs_combined.loc[i, "AUC"] = np.trapz(env_diff_lw.loc[bursts_all_limbs_combined["start"].iloc[i]:bursts_all_limbs_combined["end"].iloc[i]]) 
        + np.trapz(env_diff_rw.loc[bursts_all_limbs_combined["start"].iloc[i]:bursts_all_limbs_combined["end"].iloc[i]]) 
        + np.trapz(env_diff_ll.loc[bursts_all_limbs_combined["start"].iloc[i]:bursts_all_limbs_combined["end"].iloc[i]]) 
        + np.trapz(env_diff_rl.loc[bursts_all_limbs_combined["start"].iloc[i]:bursts_all_limbs_combined["end"].iloc[i]]) 
        + np.trapz(env_diff_trunk.loc[bursts_all_limbs_combined["start"].iloc[i]:bursts_all_limbs_combined["end"].iloc[i]]) 

    bursts_both_wrists["AUC"] = np.nan
    for i, b in enumerate(range(len(bursts_both_wrists))):
        bursts_both_wrists.loc[i, "AUC"] = np.trapz(env_diff_lw.loc[bursts_both_wrists["start"].iloc[i]:bursts_both_wrists["end"].iloc[i]]) 
        + np.trapz(env_diff_rw.loc[bursts_both_wrists["start"].iloc[i]:bursts_both_wrists["end"].iloc[i]])

    bursts_both_legs["AUC"] = np.nan
    for i, b in enumerate(range(len(bursts_both_legs))):
        bursts_both_legs.loc[i, "AUC"] = np.trapz(env_diff_ll.loc[bursts_both_legs["start"].iloc[i]:bursts_both_legs["end"].iloc[i]]) 
        + np.trapz(env_diff_rl.loc[bursts_both_legs["start"].iloc[i]:bursts_both_legs["end"].iloc[i]])

    # Trunk - I need xyz
    ax_data['trunk'].index = pd.to_datetime(ax_data['trunk']['time'], unit='s') + pd.Timedelta(hours = 1)
    ax_data['trunk'].drop(columns=['time'], inplace=True)
    trunk_acc_df = ax_data['trunk'].loc[start_sleep:end_sleep]
    del ax_data

    phi, theta = compute_spherical_coordinates(trunk_acc_df.resample('10s').median())
    trunk_acc_sph = pd.DataFrame({"phi": phi * 180 / np.pi, "theta": theta * 180 / np.pi}, index=trunk_acc_df.resample('10s').median().index)
    updated_df = detect_posture_changes(trunk_acc_sph.copy())
    time_posture_change30 = updated_df[updated_df['posture_change30']].index
    time_posture_change10 = updated_df[updated_df['posture_change10']].index
    # join bursts from all limbs and posture changes

    bursts_all_limbs_combined["posture_change"] = np.nan

    for time in time_posture_change10:
        for i in range(len(bursts_all_limbs_combined)):
                if time > bursts_all_limbs_combined["start"].iloc[i]-pd.Timedelta(seconds = 5) and time < bursts_all_limbs_combined["end"].iloc[i]+pd.Timedelta(seconds = 5):
                    bursts_all_limbs_combined["posture_change"].iloc[i] = updated_df.loc[time, "posture_change_degrees10"]
    # join bursts and posture changes

    bursts_lw["posture_change"] = np.nan
    bursts_rw["posture_change"] = np.nan
    bursts_ll["posture_change"] = np.nan
    bursts_rl["posture_change"] = np.nan
    bursts_trunk["posture_change"] = np.nan

    for time in time_posture_change30:
        for i in range(len(bursts_lw)):
            if time > bursts_lw["start"].iloc[i]-pd.Timedelta(seconds = 5) and time < bursts_lw["end"].iloc[i]+pd.Timedelta(seconds = 5):
                bursts_lw["posture_change"].iloc[i] = updated_df.loc[time, "posture_change_degrees30"]
        for i in range(len(bursts_rw)):
            if time > bursts_rw["start"].iloc[i]-pd.Timedelta(seconds = 5) and time < bursts_rw["end"].iloc[i]+pd.Timedelta(seconds = 5):
                bursts_rw["posture_change"].iloc[i] = updated_df.loc[time, "posture_change_degrees30"]
        for i in range(len(bursts_ll)):
            if time > bursts_ll["start"].iloc[i]-pd.Timedelta(seconds = 5) and time < bursts_ll["end"].iloc[i]+pd.Timedelta(seconds = 5):
                bursts_ll["posture_change"].iloc[i] = updated_df.loc[time, "posture_change_degrees30"]
        for i in range(len(bursts_rl)):
            if time > bursts_rl["start"].iloc[i]-pd.Timedelta(seconds = 5) and time < bursts_rl["end"].iloc[i]+pd.Timedelta(seconds = 5):
                bursts_rl["posture_change"].iloc[i] = updated_df.loc[time, "posture_change_degrees30"]
        for i in range(len(bursts_trunk)):
            if time > bursts_trunk["start"].iloc[i]-pd.Timedelta(seconds = 5) and time < bursts_trunk["end"].iloc[i]+pd.Timedelta(seconds = 5):
                bursts_trunk["posture_change"].iloc[i] = updated_df.loc[time, "posture_change_degrees30"]

    for time in time_posture_change10:
        for i in range(len(bursts_lw)):
            if time > bursts_lw["start"].iloc[i]-pd.Timedelta(seconds = 5) and time < bursts_lw["end"].iloc[i]+pd.Timedelta(seconds = 5):
                bursts_lw["posture_change"].iloc[i] = updated_df.loc[time, "posture_change_degrees10"]
        for i in range(len(bursts_rw)):
            if time > bursts_rw["start"].iloc[i]-pd.Timedelta(seconds = 5) and time < bursts_rw["end"].iloc[i]+pd.Timedelta(seconds = 5):
                bursts_rw["posture_change"].iloc[i] = updated_df.loc[time, "posture_change_degrees10"]
        for i in range(len(bursts_ll)):
            if time > bursts_ll["start"].iloc[i]-pd.Timedelta(seconds = 5) and time < bursts_ll["end"].iloc[i]+pd.Timedelta(seconds = 5):
                bursts_ll["posture_change"].iloc[i] = updated_df.loc[time, "posture_change_degrees10"]
        for i in range(len(bursts_rl)):
            if time > bursts_rl["start"].iloc[i]-pd.Timedelta(seconds = 5) and time < bursts_rl["end"].iloc[i]+pd.Timedelta(seconds = 5):
                bursts_rl["posture_change"].iloc[i] = updated_df.loc[time, "posture_change_degrees10"]
        for i in range(len(bursts_trunk)):
            if time > bursts_trunk["start"].iloc[i]-pd.Timedelta(seconds = 5) and time < bursts_trunk["end"].iloc[i]+pd.Timedelta(seconds = 5):
                bursts_trunk["posture_change"].iloc[i] = updated_df.loc[time, "posture_change_degrees10"]

    # summarize all the bursts in a dict, with a key for each combination of limbs

    bursts = {
        "lw": bursts_lw,
        "rw": bursts_rw,
        "ll": bursts_ll,
        "rl": bursts_rl,
        "trunk": bursts_trunk,
        "wrists": bursts_wrists_isolated,
        "legs": bursts_legs_isolated,
        "trunk_isolated": bursts_trunk_isolated,
        "both_wrists": bursts_both_wrists,
        "both_legs": bursts_both_legs,
        "all_limbs": bursts_all_limbs_combined
    }

    # SAVE
    with open(f'/Volumes/Untitled/rehab/data/{sub}/bursts_TIB.pkl', 'wb') as f:
        pickle.dump(bursts, f)
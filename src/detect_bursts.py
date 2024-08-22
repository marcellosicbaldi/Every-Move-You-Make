# Description: This script detects bursts in the accelerometer data of the trunk and limbs of the subjects, and save the results in a pickle file. 
# The bursts are detected using the envelope method, and the isolated movements are extracted for each limb. 
# The script also computes the area under the curve of the envelope for each burst, and detects posture changes from the trunk accelerometer data. 
# The results are saved in a dictionary with keys for each combination of limbs.

import numpy as np
import pandas as pd
import pickle

from functions.bursts import detect_bursts
from functions.posture import compute_spherical_coordinates, detect_posture_changes


subjects = ["158", "098", "633", "906", "279", "547", "971", "958", "815", "127", "914", "965"]

diary_SPT = {    
    "158": [pd.Timestamp('2024-02-28 23:00:00'), pd.Timestamp('2024-02-29 07:15:00')], # 158 OK
    "633": [pd.Timestamp('2024-03-07 00:05:00'), pd.Timestamp('2024-03-07 06:36:00')], # 633 OK
    "906": [pd.Timestamp('2024-03-07 00:30:00'), pd.Timestamp('2024-03-07 07:30:00')], # 906 OK
    "958": [pd.Timestamp('2024-03-13 22:00:00'), pd.Timestamp('2024-03-14 06:00:00')], # 958 OK
    "127": [pd.Timestamp('2024-03-13 23:15:00'), pd.Timestamp('2024-03-14 06:50:00')], # 127 OK
    "098": [pd.Timestamp('2024-03-16 02:03:00'), pd.Timestamp('2024-03-16 09:50:00')], # 098 OK
    "547": [pd.Timestamp('2024-03-16 01:04:00'), pd.Timestamp('2024-03-16 07:40:00')], # 547 OK
    "815": [pd.Timestamp('2024-03-20 23:00:00'), pd.Timestamp('2024-03-21 07:30:00')], # 815 OK
    "914": [pd.Timestamp('2024-03-20 21:50:00'), pd.Timestamp('2024-03-21 05:50:00')], # 914 OK
    "971": [pd.Timestamp('2024-03-20 23:50:00'), pd.Timestamp('2024-03-21 07:50:00')], # 971 OK
    "279": [pd.Timestamp('2024-03-28 00:10:00'), pd.Timestamp('2024-03-28 07:27:00')], # 279 OK
    "965": [pd.Timestamp('2024-03-28 01:25:00'), pd.Timestamp('2024-03-28 09:20:00')], # 965 OK
}

TH_WRIST = 20
TH_ANKLE = 15
TH_TRUNK = 15

for i, sub in enumerate(subjects):

    print(sub)

    start_sleep, end_sleep = diary_SPT[sub]

    trunk_df = pd.read_pickle(f"/Volumes/Untitled/rehab/data/{sub}/acc_night/trunk.pkl") * 1000
    ll_df = pd.read_pickle(f"/Volumes/Untitled/rehab/data/{sub}/acc_night/ll.pkl") * 1000
    rl_df = pd.read_pickle(f"/Volumes/Untitled/rehab/data/{sub}/acc_night/rl.pkl") * 1000
    lw_df = pd.read_pickle(f"/Volumes/Untitled/rehab/data/{sub}/acc_night/lw.pkl") * 1000
    rw_df = pd.read_pickle(f"/Volumes/Untitled/rehab/data/{sub}/acc_night/rw.pkl") * 1000
    trunk_acc_df = pd.read_pickle(f"/Volumes/Untitled/rehab/data/{sub}/acc_night/trunk_acc_3axes.pkl")

    bursts_lw = detect_bursts(lw_df, resample_envelope = True, alfa = TH_WRIST)
    bursts_rw = detect_bursts(rw_df, resample_envelope = True, alfa = TH_WRIST)
    bursts_ll = detect_bursts(ll_df, resample_envelope = True, alfa = TH_ANKLE)
    bursts_rl = detect_bursts(rl_df, resample_envelope = True, alfa = TH_ANKLE)
    bursts_trunk = detect_bursts(trunk_df, resample_envelope = True, alfa = TH_TRUNK)

    ### POSTURE ###
    phi, theta = compute_spherical_coordinates(trunk_acc_df.resample('10s').median())
    trunk_acc_sph = pd.DataFrame({"phi": phi * 180 / np.pi, "theta": theta * 180 / np.pi}, index=trunk_acc_df.resample('10s').median().index)
    updated_df = detect_posture_changes(trunk_acc_sph.copy())
    time_posture_change30 = updated_df[updated_df['posture_change30']].index
    time_posture_change10 = updated_df[updated_df['posture_change10']].index
    trunk_acc_sph["position"] = "p"
    trunk_acc_sph.loc[(trunk_acc_sph["theta"] > -45) & (trunk_acc_sph["theta"] < 45), "position"] = "supine"
    trunk_acc_sph.loc[(trunk_acc_sph["theta"] > -135) & (trunk_acc_sph["theta"] < -45), "position"] = "right"
    trunk_acc_sph.loc[(trunk_acc_sph["theta"] > 45) & (trunk_acc_sph["theta"] < 135), "position"] = "left"
    trunk_acc_sph.loc[(trunk_acc_sph["theta"] > 135) | (trunk_acc_sph["theta"] < -135), "position"] = "prone"

    df_PC = pd.DataFrame({"posture_change_degrees30": updated_df[updated_df['posture_change30']]["posture_change_degrees30"]}, index = pd.to_datetime(time_posture_change30))
    df_PC["position"] = trunk_acc_sph.loc[time_posture_change30, "position"]
    to_add = pd.DataFrame({'posture_change_degrees30': np.nan, 'position': trunk_acc_sph["position"].iloc[0]}, index = pd.DatetimeIndex([updated_df.index[0]]))
    df = pd.concat([to_add, df_PC])
    df_PC['previous_position'] = df_PC['position'].shift(1)
    # Create a column representing the transition between postures
    df_PC['transition'] = df_PC['previous_position'] + '2' + df_PC['position']
    start_posture = trunk_acc_sph.loc[df_PC.index[0] - pd.Timedelta('10s')]["position"]
    df_PC.loc[df_PC.index[0], "transition"] = start_posture + '2' + df_PC.loc[df_PC.index[0], "position"]
    # We can drop the 'Next_Position' column as it's no longer needed
    df_PC.drop('previous_position', axis=1, inplace=True)

    # join bursts and posture changes
    bursts_lw["posture_change"] = 0
    bursts_rw["posture_change"] = 0
    bursts_ll["posture_change"] = 0
    bursts_rl["posture_change"] = 0
    bursts_trunk["posture_change"] = 0
    bursts_lw["transition"] = "None"
    bursts_rw["transition"] = "None"
    bursts_ll["transition"] = "None"
    bursts_rl["transition"] = "None"
    bursts_trunk["transition"] = "None"

    for time in time_posture_change30:
        for i in range(len(bursts_lw)):
            if time > bursts_lw["start"].iloc[i]-pd.Timedelta(seconds = 5) and time < bursts_lw["end"].iloc[i]+pd.Timedelta(seconds = 5):
                bursts_lw["posture_change"].iloc[i] = updated_df.loc[time, "posture_change_degrees30"]
                bursts_lw["transition"].iloc[i] = df_PC.loc[time, "transition"]
        for i in range(len(bursts_rw)):
            if time > bursts_rw["start"].iloc[i]-pd.Timedelta(seconds = 5) and time < bursts_rw["end"].iloc[i]+pd.Timedelta(seconds = 5):
                bursts_rw["posture_change"].iloc[i] = updated_df.loc[time, "posture_change_degrees30"]
                bursts_rw["transition"].iloc[i] = df_PC.loc[time, "transition"]
        for i in range(len(bursts_ll)):
            if time > bursts_ll["start"].iloc[i]-pd.Timedelta(seconds = 5) and time < bursts_ll["end"].iloc[i]+pd.Timedelta(seconds = 5):
                bursts_ll["posture_change"].iloc[i] = updated_df.loc[time, "posture_change_degrees30"]
                bursts_ll["transition"].iloc[i] = df_PC.loc[time, "transition"]
        for i in range(len(bursts_rl)):
            if time > bursts_rl["start"].iloc[i]-pd.Timedelta(seconds = 5) and time < bursts_rl["end"].iloc[i]+pd.Timedelta(seconds = 5):
                bursts_rl["posture_change"].iloc[i] = updated_df.loc[time, "posture_change_degrees30"]
                bursts_rl["transition"].iloc[i] = df_PC.loc[time, "transition"]
        for i in range(len(bursts_trunk)):
            if time > bursts_trunk["start"].iloc[i]-pd.Timedelta(seconds = 5) and time < bursts_trunk["end"].iloc[i]+pd.Timedelta(seconds = 5):
                bursts_trunk["posture_change"].iloc[i] = updated_df.loc[time, "posture_change_degrees30"]
                bursts_trunk["transition"].iloc[i] = df_PC.loc[time, "transition"]

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

    # summarize all the bursts in a dict
    bursts = {
        "lw": bursts_lw,
        "rw": bursts_rw,
        "ll": bursts_ll,
        "rl": bursts_rl,
        "trunk": bursts_trunk
    }

    # SAVE
    with open(f'/Volumes/Untitled/rehab/data/{sub}/bursts_FINAL_envInterp_p2p.pkl', 'wb') as f:
        pickle.dump(bursts, f)
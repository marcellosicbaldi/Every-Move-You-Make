import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#### Functions to detect bursts in acceleration signal ####

def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Compute high and low envelopes of a signal s
    Parameters
    ----------
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases

    Returns
    -------
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min      
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1 
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 
    
    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s) 
        # pre-sorting of locals min based on relative position with respect to s_mid 
        lmin = lmin[s[lmin]<s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid 
        lmax = lmax[s[lmax]>s_mid]

    # global min of dmin-chunks of locals min 
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    # global max of dmax-chunks of locals max 
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
    return lmin,lmax

def detect_bursts(acc, envelope = True, plot = False, alfa = 15):
    """
    Detect bursts in acceleration signal

    Parameters
    ----------
    std_acc : pd.Series
        Standard deviation of acceleration signal with a 1 s resolution
    envelope : bool, optional
        If True, detect bursts based on the envelope of the signal
        If False, detect bursts based on the std of the signal

    Returns
    -------
    bursts : pd.Series
        pd.DataFrame with burst start times, end times, and duration
    """

    if envelope:
        lmin, lmax = hl_envelopes_idx(acc.values, dmin=9, dmax=9)
        # adjust shapes
        if len(lmin) > len(lmax):
            lmin = lmin[:-1]
        if len(lmax) > len(lmin):
            lmax = lmax[1:]
        th = np.percentile(acc.values[lmax] - acc.values[lmin], 10) * alfa
        std_acc = pd.Series(acc.values[lmax] - acc.values[lmin], index = acc.index[lmax])
    else:
        std_acc = acc.resample("1 s").std()
        std_acc.index.round("1 s")
        th = np.percentile(std_acc, 10) * alfa

    if plot:
        plt.figure()
        plt.plot(std_acc, color = 'b')
        plt.axhline(th, color = 'r')

    bursts1 = (std_acc > th).astype(int)
    start_burst = bursts1.where(bursts1.diff()==1).dropna()
    end_burst = bursts1.where(bursts1.diff()==-1).dropna()
    if bursts1.iloc[0] == 1:
            start_burst = pd.concat([pd.Series(0, index = [bursts1.index[0]]), start_burst])
    if bursts1.iloc[-1] == 1:
        end_burst = pd.concat([end_burst, pd.Series(0, index = [bursts1.index[-1]])])
    bursts_df = pd.DataFrame({"duration": end_burst.index - start_burst.index}, index = start_burst.index)

    start = bursts_df.index
    end = pd.to_datetime((bursts_df.index + bursts_df["duration"]).values)

    end = end.to_series().reset_index(drop = True)
    start = start.to_series().reset_index(drop = True)

    duration_between_bursts = (start.iloc[1:].values - end.iloc[:-1].values)

    for i in range(len(start)-1):
        if duration_between_bursts[i] < pd.Timedelta("5 s"):
            end[i] = np.nan
            start[i+1] = np.nan
    end.dropna(inplace = True)
    start.dropna(inplace = True)

    # extract amplitude of the bursts
    bursts = pd.DataFrame({"start": start.reset_index(drop = True), "end": end.reset_index(drop = True)})
    burst_amplitude1 = []
    burst_amplitude2 = []
    for i in range(len(bursts)):
        # peak-to-peak amplitude of bp acceleration
        burst_amplitude1.append(acc.loc[bursts["start"].iloc[i]:bursts["end"].iloc[i]].max() - acc.loc[bursts["start"].iloc[i]:bursts["end"].iloc[i]].min())
        # AUC of std_acc
        burst_amplitude2.append(np.trapz(std_acc.loc[bursts["start"].iloc[i]:bursts["end"].iloc[i]]))
    bursts["duration"] = bursts["end"] - bursts["start"]
    bursts["peak-to-peak"] = burst_amplitude1
    bursts["AUC"] = burst_amplitude2
    return bursts

#### Functions to filter bursts that are too close to each other ####

def filter_bursts(data):
    """
    Filter bursts that are neither preceded nor followed by another movement for at least 30 seconds.

    Parameters:
    - data (pd.DataFrame): DataFrame containing 'start', 'end', and 'duration' columns.

    Returns:
    - pd.DataFrame: Filtered DataFrame.
    """
    
    # Calculate the time difference between movements
    data['next_start_diff'] = data['start'].shift(-1) - data['end']
    data['prev_end_diff'] = data['start'] - data['end'].shift(1)
    
    # Convert differences to total seconds for comparison
    data['next_start_diff_seconds'] = data['next_start_diff'].dt.total_seconds()
    data['prev_end_diff_seconds'] = data['prev_end_diff'].dt.total_seconds()
    
    # Filter movements with at least 30 seconds separation from both previous and next movements
    filtered_data = data[(data['next_start_diff_seconds'] > 30) & (data['prev_end_diff_seconds'] > 30)]

    data.drop(columns=['next_start_diff', 'prev_end_diff', 'next_start_diff_seconds', 'prev_end_diff_seconds'], inplace=True)
    
    # Return the filtered data, dropping the temporary columns used for filtering
    return filtered_data.drop(columns=['next_start_diff', 'prev_end_diff', 'next_start_diff_seconds', 'prev_end_diff_seconds'])

#### Functions to find combination of bursts happening at different limbs ####

# For now, implemented for 
# - all 5 limbs together


def is_isolated(start, end, df):
    # Check if the start or end of an interval falls within any interval in the dataframe
    overlap = df[(df['start'] <= end) & (df['end'] >= start)]
    return overlap.empty

def merge_excluding(current_df):
    df_list = [bursts_ll, bursts_rl, bursts_lw, bursts_rw, bursts_trunk]  # TODO: make this a function argument...
    combined_df = pd.concat([df for df in df_list if not df.equals(current_df)], ignore_index=True)
    return combined_df

def find_isolated_combination(dfs_to_combine, dfs_to_isolate):
    # Merge dataframes that should be combined
    combined_df = pd.concat(dfs_to_combine, ignore_index=True).sort_values(by='start')
    # Merge dataframes from which isolation is required
    isolate_df = pd.concat(dfs_to_isolate, ignore_index=True).sort_values(by='start')

    # Finding overlaps within combined_df
    overlaps = []
    for i, row in combined_df.iterrows():
        overlapping_rows = combined_df[
            (combined_df['start'] <= row['end']) &
            (combined_df['end'] >= row['start']) &
            (combined_df.index != i)
        ]
        if not overlapping_rows.empty:
            # Check isolation from other dataframes
            if is_isolated(row['start'], row['end'], isolate_df):
                overlaps.append(row)

    return pd.DataFrame(overlaps)

def find_combined_movements_all_limbs(dfs):
    # Merging all limb dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    # Sorting by start time
    merged_df.sort_values(by='start', inplace=True)
    
    # Finding overlapping intervals for all limbs
    overlaps = []
    current_overlap = None
    for index, row in merged_df.iterrows():
        if current_overlap is None:
            current_overlap = {
                'start': row['start'],
                'end': row['end'],
                'limbs_involved': {row['limb']}
            }
        else:
            # Check if the current row overlaps with the current overlapping period
            if row['start'] <= current_overlap['end']:
                current_overlap['limbs_involved'].add(row['limb'])
                # Update the end time to the latest end time
                if row['end'] > current_overlap['end']:
                    current_overlap['end'] = row['end']
            else:
                # Check if the previous overlap involved all limbs
                if current_overlap['limbs_involved'] == {'lw', 'rw', 'll', 'rl', 'trunk'}:
                    overlaps.append(current_overlap)
                # Start a new overlap
                current_overlap = {
                    'start': row['start'],
                    'end': row['end'],
                    'limbs_involved': {row['limb']}
                }
    
    # Final check at the end of the loop
    if current_overlap and current_overlap['limbs_involved'] == {'lw', 'rw', 'll', 'rl', 'trunk'}:
        overlaps.append(current_overlap)
    
    return pd.DataFrame(overlaps)
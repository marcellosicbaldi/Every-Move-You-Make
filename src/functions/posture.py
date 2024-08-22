import numpy as np

def compute_spherical_coordinates(acc):
    acc_norm = acc.iloc[:, 0]**2 + acc.iloc[:, 1]**2 + acc.iloc[:, 2]**2
    phi = np.arccos(acc.iloc[:, 2] / acc_norm)
    theta = np.arctan2(acc.iloc[:, 1], acc.iloc[:, 0])
    return phi, theta

def circular_mean(angles_deg):
    angles_rad = np.deg2rad(angles_deg)
    mean_angle_rad = np.arctan2(np.mean(np.sin(angles_rad)), np.mean(np.cos(angles_rad)))
    mean_angle_deg = np.rad2deg(mean_angle_rad)
    return mean_angle_deg

def detect_posture_changes(df, window_size_samples=1, angle_threshold1=30, angle_threshold2=10):
    # Convert timestamps to seconds and calculate the window in terms of indices
    df['time_seconds'] = (df.index - df.index[0]).total_seconds()
    df.sort_values('time_seconds', inplace=True)  # Ensure data is sorted by time
    
    # Initialize the list to hold the indices of posture changes
    posture_change_indices30 = []
    posture_change_degrees30 = [] # to store the degree of change only for those exceeding 30
    posture_change_indices10 = []
    posture_change_degrees10 = [] # to store the degree of change only for those exceeding 10

    # Loop over the DataFrame with a step equal to the window size to ensure no overlap
    for start_idx in range(0, len(df) - window_size_samples, window_size_samples):
        # Define the window's end index
        end_idx = start_idx + window_size_samples
        
        # Calculate circular mean for the current window
        current_mean = circular_mean(df['theta'].iloc[start_idx:end_idx])
        
        # If not the first window, calculate and check against the preceding window's mean
        if start_idx > 0:
            # Calculate circular mean for the preceding window
            prev_start_idx = max(0, start_idx - window_size_samples)
            prev_end_idx = start_idx
            prev_mean = circular_mean(df['theta'].iloc[prev_start_idx:prev_end_idx])

            # prev_window_start_idx = start_idx - 2 * window_size_samples
            # prev_window_end_idx = start_idx - window_size_samples
            # prev_window = df['theta'].iloc[prev_window_start_idx:prev_window_end_idx]
            # prev_mean = circular_mean(prev_window)
            
            # Check if the orientation change exceeds the threshold
            orientation_change30 = np.abs(np.arctan2(np.sin(np.deg2rad(current_mean - prev_mean)), np.cos(np.deg2rad(current_mean - prev_mean))))
            orientation_change10 = np.abs(np.arctan2(np.sin(np.deg2rad(current_mean - prev_mean)), np.cos(np.deg2rad(current_mean - prev_mean))))
            if orientation_change30 > np.deg2rad(angle_threshold1):
                posture_change_indices30.append(start_idx)  # Mark the start of the window as a change point
                posture_change_degrees30.append(np.rad2deg(orientation_change30))
            if orientation_change10 > np.deg2rad(angle_threshold2):
                posture_change_indices10.append(start_idx)
                posture_change_degrees10.append(np.rad2deg(orientation_change10))

    # Marking boundaries for posture change
    datetime_indices30 = df.index.to_series().iloc[posture_change_indices30]
    df['posture_change30'] = False
    df.loc[datetime_indices30, 'posture_change30'] = True
    df['posture_change_degrees30'] = 0
    df.loc[datetime_indices30, 'posture_change_degrees30'] = posture_change_degrees30
    datetime_indices10 = df.index.to_series().iloc[posture_change_indices10]
    df['posture_change10'] = False
    df.loc[datetime_indices10, 'posture_change10'] = True
    df['posture_change_degrees10'] = 0
    df.loc[datetime_indices10, 'posture_change_degrees10'] = posture_change_degrees10

    return df
# Movement-HR-Sleep
This repo contains the code relative to the analysis on the HR response following movement bursts detected with accelerometers during sleep

It is organized in the following way: `src/functions` containes the functions used to process raw accelerometer and raw ECG data; `src/detect_bursts.py` and `src/HR_response.py` are the scripts used to produce the results

The most relevant functions are inside `src/functions/bursts.py`and `src/functions/HR_response.py`, and are:
  - `detect_bursts()`: bursts detection from raw acceleration based on signal envelopes
  - `characterize_bursts()`: Classify each burst based on the sensors involved in the movement
  - `detect_HR_change_from_RR()`: Detect HR response to burst from RR intervals 

The folder `annotations` contains two graphical user interfaces (`GUI_ecg_artifacts.py` and `GUI_bursts.py`, used to mark artifacts in the ECG time-series and to annotate accelerometer bursts for algorithm validation, respectively).

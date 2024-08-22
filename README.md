# Movement-HR-Sleep
This repo contains the code relative to the paper "The motor and cardiac landscape during sleep explored with a wearable multi-sensor setup" submitted to ...

All relevant code is inside the `src` folder. 

In particular, `src/functions` contains the functions used to process raw accelerometer and raw ECG data; `src/detect_bursts.py` and `src/HR_response.py` are the scripts used to produce the results

:::note
this is a note!
:::

The most relevant functions are inside `src/functions/bursts.py`and `src/functions/HR_response.py`, and are:
  - `detect_bursts()`: bursts detection from raw acceleration based on signal envelopes
  - `characterize_bursts()`: Classify each burst based on the sensors involved in the movement
  - `detect_HR_change_from_RR()`: Detect HR response to burst from RR intervals 

The folder `annotations` contains two graphical user interfaces (`GUI_ecg_artifacts.py` and `GUI_bursts.py`), used to mark artifacts in the ECG time-series and to annotate accelerometer bursts for algorithm validation, respectively.

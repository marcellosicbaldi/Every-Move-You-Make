# Movement-HR-Sleep
This repo contains the code relative to the analysis on the HR response following movement bursts detected with accelerometers during sleep

It is organized in the following way: `src/functions` containes the functions used to process raw accelerometer and raw ECG data; `src/notebook_bursts.ipynb` and `src/notebook_HRresponse.ipynb` are the notebooks used to produce the results

The most relevant functions are inside `src/functions/bursts.py`and `src/functions/HR_response.py`, and are:
  - `detect_bursts()`: bursts detection from raw acceleration based on signal envelopes
  - `characterize_bursts()`: Classify each burst based on the sensors involved in the movement
  - `detect_HR_change_from_RR()`: Detect HR response to burst from RR intervals 

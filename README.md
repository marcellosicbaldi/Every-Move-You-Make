# Movement-HR-Sleep
This repo contains the code relative to the analysis on the HR response following movement bursts detected with accelerometers during sleep

It is organized in the following way: 

- `src/functions`contains the functions used to process the raw accelerometer and raw ECG data. The most important ones are inside the script `bursts.py`and `HR_response.py:
  - `detect_bursts()` inside `bursts.py`: bursts detection from raw acceleration based on signal envelopes
  - `characterize_bursts()` 

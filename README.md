# Movement-HR-Sleep
> Companion repository for:  
> **“Mapping the physiological landscape of body movements during nocturnal sleep and wakefulness and their cardiovascular correlates with a wearable multi-sensor array”**  
> Sicbaldi M., Di Florio P., Palmerini L., Ferri R., Chiari L., Silvani A. — *Scientific Reports* (2025).

---

## ✨ Highlights

- 🧩 **Multi-sensor setup**: 5 IMUs + PPG armband + ECG chest belt
- 🏠 **Home recordings** in healthy participants
- 🦵 Detects **segmental → regional → global** movements across body locations
- ❤️ Quantifies **movement-triggered HR & PWA dynamics** (time-locked responses)
- ✅ Movement detection **validated vs. human annotations**

---

## 🔍 What this repo is about

Sleep movements contain rich information — not just *how much* you move, but **where**, **how**, and **what happens to the cardiovascular system around those events**.

This repository supports a pipeline that:
1. Detects movements from raw accelerometry across multiple body segments  
2. Merges overlapping segmental events into composite movement types  
3. Extracts time-locked **HR** (ECG) and **PWA** (PPG) responses around movements  
4. Summarizes movement “topography” and cardiovascular signatures across sleep/wake :contentReference[oaicite:2]{index=2}

---

## 🧪 Dataset (LOOKING-GLASS MIAR)

**N = 12** healthy volunteers recorded for ~1 day including one night at home. :contentReference[oaicite:3]{index=3}

### Sensors
- **5× IMUs** (Axivity AX6, 100 Hz): both wrists, both ankles, lower back (L5)
- **PPG** (Polar Verity Sense, 55 Hz)
- **ECG** (Polar H10, 130 Hz)
- Sleep diary + actigraphy-defined sleep/wake (VH2015/GGIR on non-dominant wrist) :contentReference[oaicite:4]{index=4}

### Synchronization
Devices are aligned via a **mechanical “tap” protocol** using bursts of acceleration as time markers (plus re-sync strategies if Bluetooth reconnections occur). :contentReference[oaicite:5]{index=5}

---

## 🦿 Movement detection (high level)

For each body segment:
- compute acceleration **SVM**
- band-pass filter
- derive upper/lower envelopes
- detect bursts via **threshold crossings** of envelope difference :contentReference[oaicite:6]{index=6}

Thresholds are tuned and validated against **manual movement annotations** (two independent raters → consensus). :contentReference[oaicite:7]{index=7}

---

## 🗺️ Movement “topography”

Detected events are categorized as:
- **Segmental** (single sensor)
- **Regional** (upper body / lower body / cross-regional)
- **Global** (whole-body; also used to infer postural changes when applicable)

Overlapping events across segments are merged into a **single composite movement** with:
- duration = first onset → last offset  
- magnitude = sum of peak envelope differences across involved segments :contentReference[oaicite:8]{index=8}

---

## ❤️ HR & PWA correlates (event-locked)

For each movement:
- extract a window around onset (**pre → post**)
- baseline-normalize HR and PWA
- compute summary peaks/troughs (e.g., HR peak, PWA trough)
- optional filtering: keep **isolated movements** (no other movement within ±30 s)
- manual QC for HR/PPG artifacts (short segments interpolated; long segments dropped) :contentReference[oaicite:9]{index=9}

---

## 📌 Key takeaways (headline)

- Movements show distinct **spatial patterns** (segmental vs global) across sleep/wake.
- Movements are associated with characteristic **HR increases** and **PWA decreases**,
  with patterns that vary by movement type and sleep/wake state.
- Larger, more complex movements tend to show stronger cardiovascular responses,
  supporting the idea of **movement-related autonomic signatures** as potential digital biomarkers. :contentReference[oaicite:10]{index=10}

---

## 📁 Repository structure (suggested)

All relevant code is inside the `src` folder. 

In particular, `src/functions` contains the functions used to process raw accelerometer and raw ECG data; `src/detect_bursts.py` and `src/HR_response.py` are the scripts used to produce the results

The most relevant functions are inside `src/functions/bursts.py`and `src/functions/HR_response.py`, and are:
  - `detect_bursts()`: bursts detection from raw acceleration based on signal envelopes
  - `characterize_bursts()`: Classify each burst based on the sensors involved in the movement
  - `detect_HR_change_from_RR()`: Detect HR response to burst from RR intervals 

The folder `annotations` contains two graphical user interfaces (`GUI_ecg_artifacts.py` and `GUI_bursts.py`), used to mark artifacts in the ECG time-series and to annotate accelerometer bursts for algorithm validation, respectively.

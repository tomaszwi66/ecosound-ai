# 🌍 EcoSound AI

**Ecosystem health analysis through sound.**  
EcoSound AI records or loads environmental audio and produces a scored report on biodiversity, naturalness, and acoustic richness - no specialist hardware required.

---

## Overview

Healthy ecosystems are loud in a particular way: layered bird calls, insect choruses, flowing water, and wind.  
EcoSound AI quantifies that complexity using established bioacoustic indices (ACI, ADI, BIO, NDSI) and maps them to an overall **Ecosystem Health Score (0–100)**.

A full-colour dashboard is generated automatically for every analysis.

![Dashboard example](docs/dashboard_example.png)

---

## Features

- 🎙️ **Live recording** from any laptop or desktop microphone  
- 📂 **File analysis** - drop in any `.wav` recording  
- 📊 **Five acoustic indices**: ACI · ADI · BIO · NDSI · Spectral & Temporal Entropy  
- 🐦 **Optional bird species ID** via [BirdNET-Analyzer](https://github.com/kahst/BirdNET-Analyzer) (Cornell Lab)  
- 🗺️ **Location comparison** - rank multiple recordings side by side  
- 🖥️ **Single-window dashboard** with spectrogram, waveform, radar chart, and score bars  
- 🔁 **Synthetic demo** - no microphone needed to try it out  

---

## Acoustic Indices

| Index | Full name | What it measures |
|-------|-----------|-----------------|
| **ACI** | Acoustic Complexity Index | Variation in sound intensity - higher = more complex biotic activity |
| **ADI** | Acoustic Diversity Index | Shannon diversity across frequency bands |
| **BIO** | Bioacoustic Index | Energy in the 2–8 kHz bird-song band |
| **NDSI** | Normalized Difference Soundscape Index | Ratio of biophony to anthrophony; negative = human noise dominant |
| **H(s)** | Spectral Entropy | Uniformity of energy across frequencies |
| **H(t)** | Temporal Entropy | Uniformity of energy over time |

---

## Requirements

```
Python >= 3.9
numpy
scipy
matplotlib
soundfile
```

Optional (for live recording):
```
sounddevice
```

Optional (for bird species ID):
```
birdnet
```

---

## Installation

```bash
git clone https://github.com/tomaszwi66/ecosound-ai.git
cd ecosound-ai

pip install numpy scipy matplotlib soundfile

# Optional: microphone recording
pip install sounddevice

# Optional: bird species identification
pip install birdnet

# Linux only: GUI window support
sudo apt install python3-tk
```

---

## Usage

### Quick demo (no microphone needed)
```bash
python main.py --demo
```

### Demo - single environment
```bash
python main.py --demo --env forest    # pristine forest
python main.py --demo --env park      # suburban park
python main.py --demo --env city      # urban centre
python main.py --demo --env meadow    # meadow / wetland
```

### Analyse an existing file
```bash
python main.py --file recording.wav
python main.py --file recording.wav --save        # save dashboard to PNG
python main.py --file recording.wav --birds       # + bird species ID
```

### Record from microphone
```bash
python main.py --record                           # 30 seconds (default)
python main.py --record --duration 60 --save      # 60 s, save PNG
```

### Compare multiple locations
```bash
# Place .wav files in the samples/ directory, then:
python main.py --compare
```

### BirdNET geolocation (improves accuracy)
```bash
python main.py --file forest.wav --birds --lat 52.23 --lon 21.01
```

---

## Output

### Console
```
════════════════════════════════════════════════════════════
  🎵 Analysing: forest.wav
  ⏱  Duration: 30.0s | SR: 22050 Hz
════════════════════════════════════════════════════════════
  📐 Computing acoustic indices...
    ACI  = 0.6721
    ADI  = 1.8340
    BIO  = 87.42
    NDSI = +0.6103
    H(s) = 7.2910
    H(t) = 5.4820
    Sound events: 34

  📊 RESULTS:
    Biodiversity:      81.3/100
    Naturalness:       80.5/100
    Acoustic richness: 74.2/100
    ECOSYSTEM HEALTH:  79.6/100
  ──────────────────────────────────────────────────────
  🟡 GOOD - Healthy ecosystem with minor disturbances
  🌲 Forest / Wilderness (bird-dominated)
  🐦 Biophony (nature dominant)
```

### Dashboard (PNG or interactive window)

Three rows:
1. **Spectrogram** - power over time and frequency (magma colormap)  
2. **Waveform** - amplitude over time with detected sound events highlighted  
3. **Frequency bands** · **Radar profile** · **Health scores**

---

## Health Score Scale

| Score | Category |
|-------|----------|
| 80–100 | 🟢 **Excellent** - outstanding condition |
| 60–79  | 🟡 **Good** - healthy with minor disturbances |
| 40–59  | 🟠 **Moderate** - noticeable acoustic degradation |
| 20–39  | 🔴 **Poor** - significant degradation |
| 0–19   | ⚫ **Critical** - acoustically silent |

---

## Project Structure

```
ecosound-ai/
├── main.py              # Entry point, CLI, demo environments
├── audio_analyzer.py    # Acoustic index computation (ACI, ADI, BIO, NDSI …)
├── biodiversity.py      # Scoring, classification, warnings, recommendations
├── visualizer.py        # Dashboard rendering (matplotlib)
├── bird_identifier.py   # BirdNET wrapper (optional)
├── recorder.py          # Microphone recording (optional)
└── samples/             # Place your .wav files here for --compare
```

---

## Limitations

- Analysis quality scales with recording length; **30–120 seconds** is recommended.  
- BirdNET accuracy depends on audio quality and geographic region.  
- NDSI is sensitive to wind and rain, which can lower naturalness scores artificially.  
- The GUI window requires a desktop environment. On headless servers the dashboard is saved as a PNG automatically.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- **BirdNET-Analyzer** - Cornell Lab of Ornithology & Chemnitz University of Technology  
- Acoustic index methodology based on:  
  - Pieretti et al. (2011) - Acoustic Complexity Index  
  - Villanueva-Rivera et al. (2011) - Acoustic Diversity Index  
  - Kasten et al. (2012) - NDSI

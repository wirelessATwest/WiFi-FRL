# WiFi-FRL Simulator (WiFRLsim)
_A Federated Reinforcement Learning–based Simulator for Multi-AP Wi-Fi Networks to optimise the resources_

---

## Overview
WiFi-FRL is a Python-based simulator that reproduces the fixed, random, RL, and FRL link activation (LA) schemes for dense multi-AP Wi-Fi networks, originally developed in MATLAB.  
It models a multi-link multi-access-point (AP) environment, where each AP–STA pair dynamically selects link(s) under interference and fairness constraints.

This simulator was developed as part of the research **funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement n 945380.**

---

## Features
- Fully modular: each LA scheme runs independently  
- Supports **Fixed**, **Random**, **RL**, and **FRL (federated)** policies  
- Adjustable:
  - Number of APs / STAs
  - Simulation iterations & realizations
  - Channel width, transmit power, noise power
  - FRL fairness strategy: *mini-max*, *average*, *proportional*  
- Generates:
  - `.txt` result matrices (for each scheme)
  - Timestamped log directories (`logs/run_YYYYMMDD_HHMMSS`)
  - Publication-ready plots (`figures.py`)

---
## Repository Structure
```bash
WiFi-FRL/
├─ main.py # orchestrates all schemes sequentially
├─ config.py # global simulation parameters
├─ environment.py # AP/STA placement & pathloss model
├─ fixed_bandwidth.py # fixed LA scheme
├─ random_bandwidth.py # random LA scheme
├─ rl_agent.py # RL logic (ε-greedy + reward update)
├─ frl_agent.py # FRL logic (federated reward + fairness)
├─ run_fixed.py # wrapper for Fixed LA
├─ run_random.py # wrapper for Random LA
├─ run_rl.py # wrapper for RL LA
├─ run_frl.py # wrapper for FRL LA
├─ figures.py # visualization script
├─ utils_io.py # file I/O utilities
├─ requirements.txt
├─ .gitignore
└─ logs/ # auto-created per run (ignored by git)
```
---
## Simulation Parameters (from `config.py`)

| Parameter | Description | Default |
|------------|-------------|----------|
| `Pt` | Transmit power (dBm) | 20 |
| `Pn` | Noise power (dBm) | -96 |
| `AP_COUNTS` | Number of APs simulated | `[2, 4, 8, 12, 16]` |
| `STA_COUNTS` | Number of STAs (usually = APs) | `[2, 4, 8, 12, 16]` |
| `MAX_SIM` | Number of random realizations | 500 |
| `MAX_ITER` | Number of transmission iterations | 2000 |
| `LINKS` | Channels per AP (link aggregation) | 4 |
| `CH_WIDTH` | Channel width (MHz) | 80 |
| `RSSI` | RSSI threshold (dBm) | -82 |
| `DEPLOY_MODE` | 0=random placement, 1=fixed grid | 0 |
| `SEED` | Random seed | 100 |
| `FAIR_STRATEGY` | FRL fairness: 1=mini-max, 2=avgFair, 3=propFair | 1 |

You can change these in `config.py` before running `main.py`.

---

## How to Run

### **1. Install dependencies**
```bash
pip install -r requirements.txt
```
### **2. Run simulations**
```bash
python main.py
```
This runs Fixed, Random, RL, and FRL schemes sequentially.
Results are saved under:
```bash
logs/
└─ run_YYYYMMDD_HHMMSS/
   ├─ fixed/
   ├─ random/
   ├─ rl/
   └─ frl/
```
Each folder contains:
```bash
acc...txt   # per-AP average throughput per realization
cdf...txt   # per-iteration rates for CDF and temporal analysis
```
### **3. Generate figures**
```bash
python figures.py
```

- Detects the latest run automatically.
  
- Creates figures/ inside the run folder with: CDF plots, Temporal evolution plots, Per-AP bar charts, Network density vs. throughput curve

You can specify a particular run or fairness type:
```bash
python figures.py --logdir logs/run_20251103_120501 --fair 2
```
**Data & Reproducibility**

The simulator produces all numerical datasets on-the-fly under logs/.

## Environment Requirements

| Component | Version / Notes |
|-----------|-----------------|
| Python | ≥3.9 |
| NumPy | ≥1.24 |
| Matplotlib | ≥3.7 |
| OS | Windows / Linux |
| CPU/GPU | CPU simulation (no GPU dependency) |

Optional GPU acceleration (e.g., PyTorch) can be added for future deep RL extensions.

## Reference
If you use this simulator, please cite:
- R. Ali and B. Bellalta, "A Federated Reinforcement Learning Framework for Link Activation in Multi-Link Wi-Fi Networks," 2023 IEEE International Black Sea Conference on Communications and Networking (BlackSeaCom), Istanbul, Turkiye, 2023, pp. 360-365, doi: 10.1109/BlackSeaCom58138.2023.10299778.

## Acknowledgements
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No. 945380.

## License
Released under the MIT License.
You are free to use, modify, and distribute this code with proper attribution.

© 2025 Rashid Ali — University West, Sweden  
Contact: rashid.ali@hv.se

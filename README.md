# SpeedEstimator — MiDaS × YOLOv8 × ByteTrack × Kalman Filter

**Final Project — Image Processing and Artificial Intelligence Applications**

By Max Shen, Reynard Wijaya, Diether S. Keh (SID: 1123534, 1111549, 1123802 respectively)

A hybrid vehicle speed estimation pipeline that fuses monocular depth (MiDaS DPT-Large), real-time object detection (YOLOv8), multi-object tracking (ByteTrack + Kalman Filter), and optional virtual-gate geometry to estimate per-vehicle speed from a single camera.

---

## Repository structure

```
├── src/
│   ├── cell1_install.py        # Dependency installer + CUDA check
│   ├── cell2_calibration.py    # Camera calibration & virtual-gate picker
│   └── speed_estimator.py      # Main hybrid pipeline
├── data/
│   └── README.md               # Dataset instructions (footage not included)
├── paper/
│   └── vehicle_speed_paper.pdf # Final report
├── requirements.txt
└── README.md
```

---

## Quick start

### 1 — Install dependencies

```bash
pip install -r requirements.txt
# OR run the guided installer:
python src/cell1_install.py
```

Requires Python ≥ 3.9 and PyTorch with CUDA (GPU strongly recommended for MiDaS).

### 2 — Calibrate the camera / set virtual gate

```bash
python src/cell2_calibration.py --video data/scene.mp4 --gate_real_m 3.5 --output data/calib.yaml
```

Click 4 points in the pop-up window: 2 for the top gate line, 2 for the bottom gate line.
`--gate_real_m` is the real-world distance between those two lines (e.g. one lane width ≈ 3.5 m).

### 3 — Run the speed estimator

```bash
# Full hybrid mode (MiDaS + gate geometry)
python src/speed_estimator.py --video data/scene.mp4 --calib data/calib.yaml --mode full

# Gate geometry only (no depth)
python src/speed_estimator.py --video data/scene.mp4 --calib data/calib.yaml --mode geo_only

# MiDaS depth only (no gate)
python src/speed_estimator.py --video data/scene.mp4 --calib data/calib.yaml --mode no_gate

# Save annotated output video
python src/speed_estimator.py --video data/scene.mp4 --calib data/calib.yaml --mode full --output out.mp4
```

Press **Q** to quit early. At the end of each run, a JSON block is printed with experiment metrics (mean speed, std dev, phantom rate) for Table I of the paper.

---

## Pipeline overview

```
Video frame
    │
    ├──► YOLOv8n  ──────────────────► vehicle bounding boxes
    │                                         │
    ├──► MiDaS DPT-Large (every 3rd frame)   │
    │         │                               │
    │    inverse depth map                    │
    │         │                               ▼
    │         └──────────────────► ByteTracker (IoU match + Kalman KF)
    │                                         │
    │                               per-track history
    │                                         │
    │                          ┌──────────────┴──────────────┐
    │                     gate crossing                  MiDaS disp
    │                     speed (m/s)                   speed (m/s)
    │                          └──────────────┬──────────────┘
    │                                  weighted fusion
    │                                   (0.6 gate + 0.4 depth)
    │                                         │
    └──────────────────────────────► km/h annotation on frame
```

---

## Ablation modes (Table I)

| `--mode` | Depth source | Gate geometry | Notes |
|----------|-------------|---------------|-------|
| `full`   | MiDaS DPT-Large | ✓ | Best accuracy |
| `geo_only` | — | ✓ | Fast, needs calibrated gate |
| `no_gate`  | MiDaS DPT-Large | — | No calibration needed |

---

## References

1. **YOLOv8**: Jocher, G. et al. *Ultralytics YOLOv8*, 2023. https://github.com/ultralytics/ultralytics  
2. **MiDaS**: Ranftl, R. et al. *Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer*, TPAMI 2022.  
3. **ByteTrack**: Zhang, Y. et al. *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*, ECCV 2022.  
4. **Kalman Filter**: Welch, G. & Bishop, G. *An Introduction to the Kalman Filter*, UNC Chapel Hill, 2006.  

---

## License

This project is released for academic use. See individual model licenses for YOLOv8 (AGPL-3.0) and MiDaS (MIT).

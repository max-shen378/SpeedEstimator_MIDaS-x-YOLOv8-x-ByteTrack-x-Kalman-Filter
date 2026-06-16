# Dataset / Test Footage

## Overview

This directory is intentionally left empty in the repository — video files are too large for Git.

## What you need

| File | Description |
|------|-------------|
| `scene.mp4` | Your traffic/road footage for speed estimation |
| `calib.yaml` | Output from `cell2_calibration.py` (camera intrinsics + gate coords) |

## Obtaining test footage

You can use:
- Your own dashcam or surveillance footage
- [UA-DETRAC](https://detrac-db.rit.albany.edu/) — public multi-vehicle tracking dataset
- [CityFlow](https://www.aicitychallenge.org/) — multi-camera vehicle tracking benchmark
- [MIO-TCD](https://tcd.miovision.com/) — traffic camera classification/detection dataset

## Running calibration on your footage

```bash
# Generates calib.yaml in the current directory
python src/cell2_calibration.py --video data/scene.mp4 --gate_real_m 3.5 --output data/calib.yaml
```

If you have a chessboard calibration video, add `--board`.

## File format

Place files here and reference them from the command line:

```bash
python src/speed_estimator.py --video data/scene.mp4 --calib data/calib.yaml --mode full
```

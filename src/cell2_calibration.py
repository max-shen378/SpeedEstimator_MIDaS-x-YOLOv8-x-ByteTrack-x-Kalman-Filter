"""
Cell 2 — Camera calibration & virtual gate setup.

Usage:
    python cell2_calibration.py --video <path> --output calib.yaml

Produces a YAML file with:
  - camera intrinsics (fx, fy, cx, cy, distortion)
  - virtual gate coordinates (px_top, px_bot, real_dist_m)
  - pixels-per-metre scale for the gate region

If you already have calibration params, edit calib.yaml directly
instead of running this script.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Chessboard calibration
# ---------------------------------------------------------------------------

def calibrate_from_chessboard(video_path: str, board_size=(9, 6), square_m=0.025):
    """Extract intrinsics from a chessboard calibration video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_m

    obj_pts, img_pts = [], []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % 10 != 0:   # sample every 10th frame
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, board_size)
        if found:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_pts.append(objp)
            img_pts.append(corners2)

    cap.release()
    if len(obj_pts) < 5:
        raise RuntimeError("Not enough chessboard frames found (need ≥ 5). "
                           "Check board_size or use a different video.")

    h, w = frame.shape[:2]
    ret, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, (w, h), None, None)
    print(f"Calibration RMS: {ret:.4f} px  (frames used: {len(obj_pts)})")
    return K, dist


# ---------------------------------------------------------------------------
# Interactive virtual-gate picker
# ---------------------------------------------------------------------------

_gate_points: list = []


def _mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        _gate_points.append((x, y))
        print(f"  Point {len(_gate_points)}: ({x}, {y})")


def pick_gate_from_video(video_path: str):
    """Let the user click 4 points that define a known real-world distance."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise FileNotFoundError(f"Cannot read first frame: {video_path}")

    print("\n--- Virtual gate picker ---")
    print("Click 2 points for the TOP edge of the gate, then 2 for the BOTTOM edge.")
    print("Press any key when done.")

    cv2.namedWindow("Gate picker")
    cv2.setMouseCallback("Gate picker", _mouse_cb)
    cv2.imshow("Gate picker", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(_gate_points) < 4:
        raise RuntimeError("Need 4 gate points.")

    top_mid = np.mean(_gate_points[:2], axis=0)
    bot_mid = np.mean(_gate_points[2:], axis=0)
    return tuple(map(int, top_mid)), tuple(map(int, bot_mid))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to calibration / scene video")
    ap.add_argument("--output", default="calib.yaml")
    ap.add_argument("--gate_real_m", type=float, default=3.5,
                    help="Real-world distance between gate top and bottom in metres")
    ap.add_argument("--board", action="store_true",
                    help="Run chessboard calibration instead of identity")
    args = ap.parse_args()

    if args.board:
        K, dist = calibrate_from_chessboard(args.video)
    else:
        cap = cv2.VideoCapture(args.video)
        ret, frame = cap.read()
        cap.release()
        h, w = frame.shape[:2]
        fx = fy = max(w, h)
        K = np.array([[fx, 0, w / 2],
                      [0, fy, h / 2],
                      [0,  0,      1]], dtype=float)
        dist = np.zeros(5)
        print("Using identity intrinsics (no chessboard).")

    gate_top, gate_bot = pick_gate_from_video(args.video)
    px_dist = float(np.linalg.norm(np.array(gate_top) - np.array(gate_bot)))
    ppm = px_dist / args.gate_real_m

    calib = {
        "intrinsics": {
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "dist": dist.tolist(),
        },
        "gate": {
            "top_px": list(gate_top),
            "bot_px": list(gate_bot),
            "real_dist_m": args.gate_real_m,
            "pixels_per_metre": round(ppm, 4),
        },
    }

    with open(args.output, "w") as f:
        yaml.dump(calib, f, default_flow_style=False)
    print(f"\nCalibration saved → {args.output}")
    print(f"  Gate px distance : {px_dist:.1f} px")
    print(f"  Real distance    : {args.gate_real_m} m")
    print(f"  Pixels/metre     : {ppm:.2f}")


if __name__ == "__main__":
    main()

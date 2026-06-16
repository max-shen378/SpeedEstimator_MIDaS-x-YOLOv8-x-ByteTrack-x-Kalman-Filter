"""
Hybrid Vehicle Speed Estimator
YOLOv8 × MiDaS × ByteTrack × Kalman Filter

Usage:
    python speed_estimator.py --video <path> [--calib calib.yaml]
                              [--mode full|geo_only|no_gate]
                              [--output output.mp4]

EXPERIMENT_MODE controls Table-I ablations:
  full      — MiDaS depth + virtual-gate geo correction (default)
  geo_only  — virtual-gate geometry only, no depth
  no_gate   — MiDaS depth only, no virtual gate
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# ByteTrack — lightweight re-implementation
# ---------------------------------------------------------------------------

class STrack:
    _id_counter = 0

    def __init__(self, tlwh: np.ndarray, score: float):
        STrack._id_counter += 1
        self.track_id = STrack._id_counter
        self.tlwh = tlwh.copy()
        self.score = score
        self.is_activated = True
        self.state = "tracked"
        self._kf = self._make_kf(tlwh)
        self.history: List[np.ndarray] = [tlwh.copy()]

    @staticmethod
    def _make_kf(tlwh: np.ndarray) -> KalmanFilter:
        kf = KalmanFilter(dim_x=8, dim_z=4)
        kf.F = np.eye(8)
        for i in range(4):
            kf.F[i, i + 4] = 1.0
        kf.H = np.eye(4, 8)
        kf.R *= 10.0
        kf.P[4:, 4:] *= 1000.0
        kf.Q[4:, 4:] *= 0.01
        cx, cy = tlwh[0] + tlwh[2] / 2, tlwh[1] + tlwh[3] / 2
        kf.x[:4] = np.array([[cx], [cy], [tlwh[2]], [tlwh[3]]])
        return kf

    def predict(self):
        self._kf.predict()

    def update(self, tlwh: np.ndarray, score: float):
        cx, cy = tlwh[0] + tlwh[2] / 2, tlwh[1] + tlwh[3] / 2
        z = np.array([[cx], [cy], [tlwh[2]], [tlwh[3]]])
        self._kf.update(z)
        x = self._kf.x
        w, h = float(x[2]), float(x[3])
        self.tlwh = np.array([float(x[0]) - w / 2, float(x[1]) - h / 2, w, h])
        self.score = score
        self.history.append(self.tlwh.copy())

    @property
    def tlbr(self) -> np.ndarray:
        t = self.tlwh.copy()
        t[2:] += t[:2]
        return t

    def centre(self) -> np.ndarray:
        return np.array([self.tlwh[0] + self.tlwh[2] / 2,
                         self.tlwh[1] + self.tlwh[3] / 2])


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


class ByteTracker:
    def __init__(self, iou_thr: float = 0.3, max_lost: int = 30):
        self.tracks: List[STrack] = []
        self.lost: List[STrack] = []
        self.iou_thr = iou_thr
        self.max_lost = max_lost
        self._lost_count: Dict[int, int] = defaultdict(int)

    def update(self, detections: np.ndarray) -> List[STrack]:
        """detections: Nx5 [x1,y1,x2,y2,score]"""
        for t in self.tracks:
            t.predict()

        matched_track, matched_det = set(), set()
        if self.tracks and len(detections):
            ious = np.zeros((len(self.tracks), len(detections)))
            for i, t in enumerate(self.tracks):
                for j, d in enumerate(detections):
                    ious[i, j] = _iou(t.tlbr, d[:4])
            while True:
                i, j = np.unravel_index(ious.argmax(), ious.shape)
                if ious[i, j] < self.iou_thr:
                    break
                tlwh = np.array([detections[j, 0], detections[j, 1],
                                  detections[j, 2] - detections[j, 0],
                                  detections[j, 3] - detections[j, 1]])
                self.tracks[i].update(tlwh, float(detections[j, 4]))
                matched_track.add(i)
                matched_det.add(j)
                ious[i, :] = -1
                ious[:, j] = -1

        new_tracks = []
        for i, t in enumerate(self.tracks):
            if i in matched_track:
                self._lost_count[t.track_id] = 0
                new_tracks.append(t)
            else:
                self._lost_count[t.track_id] += 1
                if self._lost_count[t.track_id] <= self.max_lost:
                    self.lost.append(t)

        for j, d in enumerate(detections):
            if j not in matched_det:
                tlwh = np.array([d[0], d[1], d[2] - d[0], d[3] - d[1]])
                new_tracks.append(STrack(tlwh, float(d[4])))

        self.tracks = new_tracks
        return self.tracks


# ---------------------------------------------------------------------------
# MiDaS depth
# ---------------------------------------------------------------------------

class MiDaSDepth:
    def __init__(self, model_type: str = "DPT_Large", device: Optional[torch.device] = None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available()
                                  else torch.device("cpu"))
        self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        self.model.to(self.device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = (transforms.dpt_transform if "DPT" in model_type
                          else transforms.small_transform)

    @torch.no_grad()
    def infer(self, bgr_frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        inp = self.transform(rgb).to(self.device)
        pred = self.model(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=bgr_frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        return pred.cpu().numpy()


# ---------------------------------------------------------------------------
# Virtual gate speed (geometry-based)
# ---------------------------------------------------------------------------

def gate_speed_mps(track: STrack, gate_top_y: int, gate_bot_y: int,
                   real_dist_m: float, fps: float) -> Optional[float]:
    """Estimate speed using the virtual gate crossing time."""
    centres_y = [h[1] + h[3] / 2 for h in track.history]
    crossings = []
    for k in range(1, len(centres_y)):
        prev_y, cur_y = centres_y[k - 1], centres_y[k]
        for gate_y in (gate_top_y, gate_bot_y):
            if (prev_y - gate_y) * (cur_y - gate_y) < 0:
                frac = abs(prev_y - gate_y) / (abs(prev_y - gate_y) + abs(cur_y - gate_y))
                crossings.append(k - 1 + frac)
    if len(crossings) >= 2:
        dt = (crossings[-1] - crossings[0]) / fps
        return real_dist_m / dt if dt > 0 else None
    return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class SpeedEstimatorPipeline:
    VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorbike, bus, truck (COCO)

    def __init__(self, video_path: str, calib_path: Optional[str],
                 mode: str = "full", output_path: Optional[str] = None):
        self.video_path = video_path
        self.mode = mode
        self.output_path = output_path

        # Load calibration
        self.gate_top_y: Optional[int] = None
        self.gate_bot_y: Optional[int] = None
        self.real_dist_m: float = 3.5
        self.ppm: float = 1.0
        if calib_path and Path(calib_path).exists():
            with open(calib_path) as f:
                cal = yaml.safe_load(f)
            g = cal.get("gate", {})
            self.gate_top_y = g.get("top_px", [None, None])[1]
            self.gate_bot_y = g.get("bot_px", [None, None])[1]
            self.real_dist_m = g.get("real_dist_m", 3.5)
            self.ppm = g.get("pixels_per_metre", 1.0)

        # Models
        self.yolo = YOLO("yolov8n.pt")
        self.tracker = ByteTracker()
        self.midas: Optional[MiDaSDepth] = None
        if mode in ("full", "no_gate"):
            print("Loading MiDaS DPT-Large …")
            self.midas = MiDaSDepth()

        # Experiment logging
        self.speed_log: Dict[int, List[float]] = defaultdict(list)  # track_id → [km/h]
        self.phantom_count = 0
        self.total_tracks = 0

    def _estimate_speed_midas(self, track: STrack, depth_map: np.ndarray, fps: float) -> Optional[float]:
        if len(track.history) < 2:
            return None
        cx0, cy0 = track.history[-2][:2] + track.history[-2][2:] / 2
        cx1, cy1 = track.history[-1][:2] + track.history[-1][2:] / 2
        px_disp = np.linalg.norm([cx1 - cx0, cy1 - cy0])
        cx0, cy0 = int(np.clip(cx0, 0, depth_map.shape[1] - 1)), int(np.clip(cy0, 0, depth_map.shape[0] - 1))
        cx1, cy1 = int(np.clip(cx1, 0, depth_map.shape[1] - 1)), int(np.clip(cy1, 0, depth_map.shape[0] - 1))
        inv_d = (depth_map[cy0, cx0] + depth_map[cy1, cx1]) / 2.0
        scale = 1.0 / (inv_d + 1e-6)
        real_disp_m = (px_disp / self.ppm) * scale
        speed_mps = real_disp_m * fps
        return speed_mps

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(self.video_path)

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Default gate to thirds of frame height if not calibrated
        if self.gate_top_y is None:
            self.gate_top_y = h // 3
        if self.gate_bot_y is None:
            self.gate_bot_y = 2 * h // 3

        writer = None
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(self.output_path, fourcc, fps, (w, h))

        depth_map: Optional[np.ndarray] = None
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # MiDaS every 3 frames (expensive)
            if self.midas and frame_idx % 3 == 0:
                depth_map = self.midas.infer(frame)

            # YOLOv8 detection
            results = self.yolo(frame, verbose=False)[0]
            dets = []
            for box in results.boxes:
                cls = int(box.cls[0])
                if cls not in self.VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = float(box.conf[0])
                if score < 0.3:
                    continue
                dets.append([x1, y1, x2, y2, score])

            detections = np.array(dets, dtype=float) if dets else np.empty((0, 5))
            tracks = self.tracker.update(detections)
            self.total_tracks = max(self.total_tracks, len(tracks))

            for t in tracks:
                speed_kmh: Optional[float] = None

                if self.mode == "full":
                    v_gate = gate_speed_mps(t, self.gate_top_y, self.gate_bot_y,
                                            self.real_dist_m, fps)
                    v_midas = (self._estimate_speed_midas(t, depth_map, fps)
                               if depth_map is not None else None)
                    if v_gate and v_midas:
                        speed_kmh = (0.6 * v_gate + 0.4 * v_midas) * 3.6
                    elif v_gate:
                        speed_kmh = v_gate * 3.6
                    elif v_midas:
                        speed_kmh = v_midas * 3.6

                elif self.mode == "geo_only":
                    v = gate_speed_mps(t, self.gate_top_y, self.gate_bot_y,
                                       self.real_dist_m, fps)
                    speed_kmh = v * 3.6 if v else None

                elif self.mode == "no_gate":
                    v = (self._estimate_speed_midas(t, depth_map, fps)
                         if depth_map is not None else None)
                    speed_kmh = v * 3.6 if v else None

                if speed_kmh and 5 < speed_kmh < 200:
                    self.speed_log[t.track_id].append(speed_kmh)
                elif speed_kmh and speed_kmh > 200:
                    self.phantom_count += 1

                # Draw
                x1, y1, x2, y2 = map(int, t.tlbr)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID:{t.track_id}"
                if speed_kmh:
                    label += f" {speed_kmh:.1f}km/h"
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Draw gate lines
            cv2.line(frame, (0, self.gate_top_y), (w, self.gate_top_y), (0, 0, 255), 2)
            cv2.line(frame, (0, self.gate_bot_y), (w, self.gate_bot_y), (0, 0, 255), 2)
            cv2.putText(frame, f"Frame {frame_idx}  Mode:{self.mode}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            if writer:
                writer.write(frame)
            cv2.imshow("Speed Estimator", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        self._print_experiment_summary()

    def _print_experiment_summary(self):
        all_speeds = [s for speeds in self.speed_log.values() for s in speeds]
        if not all_speeds:
            print("No speed measurements recorded.")
            return
        mean_s = float(np.mean(all_speeds))
        std_s = float(np.std(all_speeds))
        n_tracks = len(self.speed_log)
        phantom_rate = self.phantom_count / max(1, n_tracks + self.phantom_count)

        summary = {
            "mode": self.mode,
            "num_tracks": n_tracks,
            "mean_speed_kmh": round(mean_s, 2),
            "std_speed_kmh": round(std_s, 2),
            "phantom_rate": round(phantom_rate, 4),
            "phantom_count": self.phantom_count,
            "total_measurements": len(all_speeds),
        }
        print("\n" + "=" * 60)
        print("EXPERIMENT RESULTS (paste into Table I)")
        print("=" * 60)
        print(json.dumps(summary, indent=2))
        print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Hybrid vehicle speed estimator")
    ap.add_argument("--video", required=True)
    ap.add_argument("--calib", default=None)
    ap.add_argument("--mode", choices=["full", "geo_only", "no_gate"], default="full",
                    help="Ablation mode: full | geo_only | no_gate")
    ap.add_argument("--output", default=None, help="Save annotated video to this path")
    args = ap.parse_args()

    pipeline = SpeedEstimatorPipeline(
        video_path=args.video,
        calib_path=args.calib,
        mode=args.mode,
        output_path=args.output,
    )
    pipeline.run()

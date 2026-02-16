#!/usr/bin/env python3
import argparse
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import cv2
from flask import Flask, jsonify

from app import (
    calibrate_from_frame,
    detect_marker_homography,
    draw,
    enrich_detections_with_world,
    filter_detections_by_px_size,
    find_detections,
    have_aruco_module,
)


@dataclass
class VisionSnapshot:
    timestamp: float
    cubes: List[Dict[str, Any]]
    marker_calibrated: bool
    marker_visible: bool
    pixels_per_cm: Optional[float]


class VisionState:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.latest_frame = None
        self.snapshot = VisionSnapshot(
            timestamp=0.0,
            cubes=[],
            marker_calibrated=False,
            marker_visible=False,
            pixels_per_cm=None,
        )
        self.h_img_to_marker_cached = None
        self.marker_corners_cached = None


def parse_args():
    parser = argparse.ArgumentParser(description="Vision server pro detekci kostek na localhost.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="HTTP host")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port")
    parser.add_argument("--camera", type=int, default=0, help="Index kamery")
    parser.add_argument("--width", type=int, default=1280, help="Sirka obrazu")
    parser.add_argument("--height", type=int, default=720, help="Vyska obrazu")
    parser.add_argument("--min-area", type=int, default=450, help="Minimalni plocha kontury v px")
    parser.add_argument("--show-window", action="store_true", help="Zobraz OpenCV okno")

    parser.add_argument("--marker-id", type=int, default=23, help="ID ArUco markeru")
    parser.add_argument("--marker-size-cm", type=float, default=5.0, help="Hrana markeru v cm")
    parser.add_argument("--aruco-dict", type=str, default="DICT_4X4_50", help="ArUco slovnik")
    parser.add_argument("--marker-origin-robot-x", type=float, default=0.0, help="X markeru v robot frame [cm]")
    parser.add_argument("--marker-origin-robot-y", type=float, default=0.0, help="Y markeru v robot frame [cm]")
    parser.add_argument(
        "--marker-to-robot-yaw-deg",
        type=float,
        default=0.0,
        help="Dalsi rotace marker->robot [deg] po mapovani X dopredu, Y doleva",
    )

    parser.add_argument("--size-min-scale", type=float, default=0.65, help="Spodni tolerance velikosti")
    parser.add_argument("--size-max-scale", type=float, default=1.80, help="Horni tolerance velikosti")
    return parser.parse_args()


def detection_to_dict(det) -> Dict[str, Any]:
    out = {
        "color": det.color,
        "side_px": float(det.side_px),
        "side_cm": float(det.side_cm) if det.side_cm is not None else None,
        "robot_x_cm": None,
        "robot_y_cm": None,
        "marker_x_cm": None,
        "marker_y_cm": None,
    }

    if det.robot_xy_cm is not None:
        out["robot_x_cm"] = float(det.robot_xy_cm[0])
        out["robot_y_cm"] = float(det.robot_xy_cm[1])
    if det.marker_xy_cm is not None:
        out["marker_x_cm"] = float(det.marker_xy_cm[0])
        out["marker_y_cm"] = float(det.marker_xy_cm[1])
    return out


def calibrate_marker_from_frame(frame, state: VisionState, args) -> bool:
    h_img_to_marker, marker_corners_img = detect_marker_homography(
        frame,
        marker_id=args.marker_id,
        marker_size_cm=args.marker_size_cm,
        aruco_dict_name=args.aruco_dict,
    )
    if h_img_to_marker is None:
        return False

    with state.lock:
        state.h_img_to_marker_cached = h_img_to_marker
        state.marker_corners_cached = marker_corners_img
    return True


def calibrate_px_from_frame(frame, state: VisionState, args) -> Optional[float]:
    val = calibrate_from_frame(frame, min_area=args.min_area)
    if val is None:
        return None
    with state.lock:
        snap = state.snapshot
        state.snapshot = VisionSnapshot(
            timestamp=snap.timestamp,
            cubes=snap.cubes,
            marker_calibrated=snap.marker_calibrated,
            marker_visible=snap.marker_visible,
            pixels_per_cm=float(val),
        )
    return float(val)


def run_detection_loop(state: VisionState, args):
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError("Kameru se nepodarilo otevrit.")

    while True:
        with state.lock:
            if not state.running:
                break

        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        detections = find_detections(frame, min_area=args.min_area)

        with state.lock:
            h_img_to_marker_cached = state.h_img_to_marker_cached
            marker_corners_cached = state.marker_corners_cached
            pixels_per_cm = state.snapshot.pixels_per_cm

        marker_visible = False
        marker_corners_img = None
        marker_calibrated = h_img_to_marker_cached is not None

        if marker_calibrated:
            _, marker_corners_img = detect_marker_homography(
                frame,
                marker_id=args.marker_id,
                marker_size_cm=args.marker_size_cm,
                aruco_dict_name=args.aruco_dict,
            )
            if marker_corners_img is not None:
                marker_visible = True
                marker_corners_cached = marker_corners_img
            elif marker_corners_cached is not None:
                marker_corners_img = marker_corners_cached

            detections = enrich_detections_with_world(detections, h_img_to_marker_cached, args)
        else:
            detections = filter_detections_by_px_size(detections, pixels_per_cm, args)

        cubes = [detection_to_dict(det) for det in detections]
        ts = time.time()

        with state.lock:
            state.latest_frame = frame.copy()
            state.marker_corners_cached = marker_corners_cached
            state.snapshot = VisionSnapshot(
                timestamp=ts,
                cubes=cubes,
                marker_calibrated=marker_calibrated,
                marker_visible=marker_visible,
                pixels_per_cm=pixels_per_cm,
            )

        if args.show_window:
            frame_to_show = frame.copy()
            draw(
                frame_to_show,
                detections,
                pixels_per_cm,
                marker_corners_img,
                marker_visible,
                marker_calibrated,
            )
            cv2.imshow("Robo Cube Vision Server", frame_to_show)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                with state.lock:
                    state.running = False
                break
            if key == ord("m"):
                if calibrate_marker_from_frame(frame, state, args):
                    print("Marker kalibrace OK (klavesa m).")
                else:
                    print("Marker kalibrace selhala (marker nenalezen).")
            if key == ord("c"):
                val = calibrate_px_from_frame(frame, state, args)
                if val is not None:
                    print(f"PX kalibrace OK: {val:.2f} px/cm")

    cap.release()
    if args.show_window:
        cv2.destroyAllWindows()


def create_flask_app(state: VisionState, args):
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({"ok": True})

    @app.get("/status")
    def status():
        with state.lock:
            snap = state.snapshot
            data = {
                "running": state.running,
                "camera_ready": state.latest_frame is not None,
                "snapshot": asdict(snap),
            }
        return jsonify(data)

    @app.get("/detect_cubes")
    def detect_cubes():
        with state.lock:
            snap = state.snapshot
            data = asdict(snap)
        return jsonify(data)

    @app.post("/calibrate_marker")
    def calibrate_marker():
        with state.lock:
            frame = None if state.latest_frame is None else state.latest_frame.copy()

        if frame is None:
            return jsonify({"ok": False, "error": "camera_not_ready"}), 409

        ok = calibrate_marker_from_frame(frame, state, args)
        if not ok:
            return jsonify({"ok": False, "error": "marker_not_found"}), 409

        return jsonify({"ok": True})

    @app.post("/calibrate_px")
    def calibrate_px():
        with state.lock:
            frame = None if state.latest_frame is None else state.latest_frame.copy()

        if frame is None:
            return jsonify({"ok": False, "error": "camera_not_ready"}), 409

        val = calibrate_px_from_frame(frame, state, args)
        if val is None:
            return jsonify({"ok": False, "error": "cube_not_found"}), 409

        return jsonify({"ok": True, "pixels_per_cm": val})

    @app.post("/shutdown")
    def shutdown():
        with state.lock:
            state.running = False
        return jsonify({"ok": True})

    return app


def run_http_server(state: VisionState, args):
    app = create_flask_app(state, args)
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=True)


def main():
    args = parse_args()

    if not have_aruco_module():
        print("Varovani: cv2.aruco neni dostupne. Nainstaluj opencv-contrib-python.")

    state = VisionState()

    http_thread = threading.Thread(target=run_http_server, args=(state, args), daemon=True)
    http_thread.start()
    print(f"Vision server bezi na http://{args.host}:{args.port}")
    if args.show_window:
        print("OpenCV ovladani: m=kalibrace markeru, c=px kalibrace, q=konec")

    try:
        run_detection_loop(state, args)
    finally:
        with state.lock:
            state.running = False


if __name__ == "__main__":
    main()

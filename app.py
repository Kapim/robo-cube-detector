#!/usr/bin/env python3
import argparse
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


CUBE_EDGE_CM = 2.5


@dataclass
class Detection:
    color: str
    box: np.ndarray
    area: float
    side_px: float
    side_cm: Optional[float] = None
    marker_xy_cm: Optional[Tuple[float, float]] = None
    robot_xy_cm: Optional[Tuple[float, float]] = None


def build_color_ranges():
    # HSV rozsahy pro barvy (OpenCV: H 0-179)
    return {
        "green": [
            (np.array([35, 70, 50]), np.array([85, 255, 255])),
        ],
        "blue": [
            (np.array([90, 80, 50]), np.array([130, 255, 255])),
        ],
        "yellow": [
            (np.array([18, 80, 80]), np.array([35, 255, 255])),
        ],
        "red": [
            (np.array([0, 80, 70]), np.array([10, 255, 255])),
            (np.array([160, 80, 70]), np.array([179, 255, 255])),
        ],
    }


def color_mask(hsv, ranges):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for low, high in ranges:
        mask |= cv2.inRange(hsv, low, high)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def approx_square(contour):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    if len(approx) < 4:
        return None
    rect = cv2.minAreaRect(contour)
    (w, h) = rect[1]
    if w <= 0 or h <= 0:
        return None
    ratio = min(w, h) / max(w, h)
    if ratio < 0.75:
        return None
    box = cv2.boxPoints(rect).astype(np.int32)
    side_px = (w + h) / 2.0
    return box, side_px


def find_detections(frame, min_area):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ranges = build_color_ranges()
    detections = []

    for color_name, color_ranges in ranges.items():
        mask = color_mask(hsv, color_ranges)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            square = approx_square(cnt)
            if square is None:
                continue
            box, side_px = square
            detections.append(
                Detection(color=color_name, box=box, area=area, side_px=side_px)
            )
    return detections


def have_aruco_module():
    return hasattr(cv2, "aruco")


def detect_marker_homography(frame, marker_id, marker_size_cm, aruco_dict_name):
    if not have_aruco_module():
        return None, None

    try:
        dict_id = getattr(cv2.aruco, aruco_dict_name)
    except AttributeError:
        raise ValueError(f"Neznamy ArUco slovnik: {aruco_dict_name}")

    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)

    # OpenCV 4.7+: ArucoDetector, starsi verze: detectMarkers
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        corners, ids, _ = detector.detectMarkers(frame)
    else:
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(frame, dictionary, parameters=params)

    if ids is None:
        return None, None

    ids = ids.flatten()
    for i, mid in enumerate(ids):
        if int(mid) != marker_id:
            continue

        marker_corners_img = corners[i][0].astype(np.float32)
        marker_corners_plane = np.array(
            [
                [0.0, 0.0],
                [marker_size_cm, 0.0],
                [marker_size_cm, marker_size_cm],
                [0.0, marker_size_cm],
            ],
            dtype=np.float32,
        )

        h_img_to_marker = cv2.getPerspectiveTransform(marker_corners_img, marker_corners_plane)
        return h_img_to_marker, marker_corners_img

    return None, None


def project_point(h_img_to_marker, pt_xy):
    src = np.array([[[float(pt_xy[0]), float(pt_xy[1])]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, h_img_to_marker)
    return float(dst[0, 0, 0]), float(dst[0, 0, 1])


def marker_to_robot(xm, ym, marker_origin_robot_x, marker_origin_robot_y, marker_to_robot_yaw_deg):
    # Zakladni mapovani marker -> robot pro yaw=0:
    # robot X dopredu, robot Y doleva, marker je pred robotem bez natoceni.
    x_base = -ym
    y_base = -xm

    yaw = math.radians(marker_to_robot_yaw_deg)
    c = math.cos(yaw)
    s = math.sin(yaw)
    xr = c * x_base - s * y_base + marker_origin_robot_x
    yr = s * x_base + c * y_base + marker_origin_robot_y
    return xr, yr


def enrich_detections_with_world(detections, h_img_to_marker, args):
    out = []
    for det in detections:
        center_px = np.mean(det.box, axis=0)
        mx, my = project_point(h_img_to_marker, center_px)

        box_plane = cv2.perspectiveTransform(
            det.box.astype(np.float32).reshape(1, 4, 2),
            h_img_to_marker,
        )[0]
        side1 = float(np.linalg.norm(box_plane[1] - box_plane[0]))
        side2 = float(np.linalg.norm(box_plane[2] - box_plane[1]))
        side_cm = (side1 + side2) / 2.0

        # Filtr na kosticky 2.5 cm uz v realnych cm (bez zavislosti na px)
        if not (args.size_min_scale * CUBE_EDGE_CM <= side_cm <= args.size_max_scale * CUBE_EDGE_CM):
            continue

        xr, yr = marker_to_robot(
            mx,
            my,
            args.marker_origin_robot_x,
            args.marker_origin_robot_y,
            args.marker_to_robot_yaw_deg,
        )

        det.side_cm = side_cm
        det.marker_xy_cm = (mx, my)
        det.robot_xy_cm = (xr, yr)
        out.append(det)

    return out


def filter_detections_by_px_size(detections, pixels_per_cm, args):
    if pixels_per_cm is None:
        return detections

    expected_px = CUBE_EDGE_CM * pixels_per_cm
    out = []
    for det in detections:
        if args.size_min_scale * expected_px <= det.side_px <= args.size_max_scale * expected_px:
            out.append(det)
    return out


def draw(
    frame,
    detections,
    pixels_per_cm,
    marker_corners_img,
    marker_visible,
    marker_calibrated,
):
    color_bgr = {
        "green": (0, 255, 0),
        "red": (0, 0, 255),
        "yellow": (0, 255, 255),
        "blue": (255, 0, 0),
    }

    if marker_corners_img is not None:
        cv2.polylines(
            frame,
            [marker_corners_img.astype(np.int32)],
            isClosed=True,
            color=(255, 255, 0),
            thickness=2,
        )

    for det in detections:
        cv2.drawContours(frame, [det.box], 0, color_bgr[det.color], 2)
        center = tuple(np.mean(det.box, axis=0).astype(int))

        if det.robot_xy_cm is not None:
            text = (
                f"{det.color} R=({det.robot_xy_cm[0]:.1f},{det.robot_xy_cm[1]:.1f})cm "
                f"S={det.side_cm:.2f}cm"
            )
        else:
            text = f"{det.color} {det.side_px:.0f}px"
            if pixels_per_cm is not None:
                text += f" ({det.side_px / pixels_per_cm:.2f}cm)"

        cv2.putText(
            frame,
            text,
            center,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color_bgr[det.color],
            2,
            cv2.LINE_AA,
        )

    hint = "q: konec | c: kalibrace px/cm | m: kalibrace markeru"
    cv2.putText(
        frame,
        hint,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if marker_calibrated and marker_visible:
        msg = "Marker kalibrovan: absolutni souradnice bezi (marker viditelny)"
    elif marker_calibrated and not marker_visible:
        msg = "Marker kalibrovan: absolutni souradnice bezi (marker docasne zakryty)"
    elif pixels_per_cm is None:
        msg = "Bez marker kalibrace: stiskni m (Aruco) nebo c (px/cm)"
    else:
        msg = f"Bez marker kalibrace: fallback {pixels_per_cm:.2f} px/cm"

    cv2.putText(
        frame,
        msg,
        (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def calibrate_from_frame(frame, min_area):
    detections = find_detections(frame, min_area=min_area)
    if not detections:
        return None
    best = max(detections, key=lambda d: d.area)
    return best.side_px / CUBE_EDGE_CM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detekce barevnych kosticek (2.5 cm) pomoci OpenCV a webkamery."
    )
    parser.add_argument("--camera", type=int, default=0, help="Index kamery (default: 0)")
    parser.add_argument("--min-area", type=int, default=450, help="Minimalni plocha kontury v px")
    parser.add_argument("--width", type=int, default=1280, help="Sirka obrazu")
    parser.add_argument("--height", type=int, default=720, help="Vyska obrazu")

    parser.add_argument("--marker-id", type=int, default=23, help="ID ArUco markeru")
    parser.add_argument(
        "--marker-size-cm",
        type=float,
        default=5.0,
        help="Skutecna delka hrany markeru v cm",
    )
    parser.add_argument(
        "--aruco-dict",
        type=str,
        default="DICT_4X4_50",
        help="Nazev ArUco slovniku (napr. DICT_4X4_50)",
    )

    parser.add_argument(
        "--marker-origin-robot-x",
        type=float,
        default=0.0,
        help="X markeru [cm] v souradnicich robota",
    )
    parser.add_argument(
        "--marker-origin-robot-y",
        type=float,
        default=0.0,
        help="Y markeru [cm] v souradnicich robota",
    )
    parser.add_argument(
        "--marker-to-robot-yaw-deg",
        type=float,
        default=0.0,
        help="Dalsi rotace marker->robot kolem Z [deg] po mapovani X dopredu, Y doleva",
    )
    parser.add_argument(
        "--size-min-scale",
        type=float,
        default=0.65,
        help="Spodni nasobek ocekavane hrany kostky pro filtr velikosti",
    )
    parser.add_argument(
        "--size-max-scale",
        type=float,
        default=1.80,
        help="Horni nasobek ocekavane hrany kostky pro filtr velikosti",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not have_aruco_module():
        print(
            "Varovani: cv2.aruco neni dostupne. Nainstaluj opencv-contrib-python pro marker kalibraci."
        )

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError("Kameru se nepodarilo otevrit.")

    pixels_per_cm = None
    h_img_to_marker_cached = None
    marker_corners_cached = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = find_detections(frame, min_area=args.min_area)

        marker_visible = False
        marker_corners_img = None
        marker_calibrated = h_img_to_marker_cached is not None

        # Marker jen zkontrolujeme kvuli statusu v UI; pro vypocet pouzivame
        # posledni rucne kalibrovanou transformaci.
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

            detections = enrich_detections_with_world(
                detections, h_img_to_marker_cached, args
            )
        else:
            detections = filter_detections_by_px_size(detections, pixels_per_cm, args)

        draw(
            frame,
            detections,
            pixels_per_cm,
            marker_corners_img,
            marker_visible,
            marker_calibrated,
        )
        cv2.imshow("Robo Cube Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            val = calibrate_from_frame(frame, min_area=args.min_area)
            if val is not None:
                pixels_per_cm = val
        if key == ord("m"):
            h_img_to_marker, marker_corners_img = detect_marker_homography(
                frame,
                marker_id=args.marker_id,
                marker_size_cm=args.marker_size_cm,
                aruco_dict_name=args.aruco_dict,
            )
            if h_img_to_marker is not None:
                h_img_to_marker_cached = h_img_to_marker
                marker_corners_cached = marker_corners_img
                print("Marker kalibrace OK: transformace obraz -> marker ulozena.")
            else:
                print("Marker kalibrace selhala: marker nebyl nalezen.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

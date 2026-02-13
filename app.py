#!/usr/bin/env python3
import argparse
from dataclasses import dataclass

import cv2
import numpy as np


CUBE_EDGE_CM = 2.5


@dataclass
class Detection:
    color: str
    box: np.ndarray
    area: float
    side_px: float


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


def find_detections(frame, min_area, pixels_per_cm):
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

            if pixels_per_cm is not None:
                expected_px = CUBE_EDGE_CM * pixels_per_cm
                if not (0.65 * expected_px <= side_px <= 1.35 * expected_px):
                    continue

            detections.append(
                Detection(color=color_name, box=box, area=area, side_px=side_px)
            )
    return detections


def draw(frame, detections, pixels_per_cm):
    color_bgr = {
        "green": (0, 255, 0),
        "red": (0, 0, 255),
        "yellow": (0, 255, 255),
        "blue": (255, 0, 0),
    }
    for det in detections:
        cv2.drawContours(frame, [det.box], 0, color_bgr[det.color], 2)
        center = tuple(np.mean(det.box, axis=0).astype(int))
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

    hint = "q: konec | c: kalibrace (jedna kosticka 2.5cm)"
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

    if pixels_per_cm is None:
        msg = "Nekalibrovano: velikostni filtr je vypnuty"
    else:
        msg = f"Kalibrace: {pixels_per_cm:.2f} px/cm (cil hrana: {CUBE_EDGE_CM} cm)"
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
    detections = find_detections(frame, min_area=min_area, pixels_per_cm=None)
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
    return parser.parse_args()


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError("Kameru se nepodarilo otevrit.")

    pixels_per_cm = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = find_detections(
            frame, min_area=args.min_area, pixels_per_cm=pixels_per_cm
        )
        draw(frame, detections, pixels_per_cm)
        cv2.imshow("Robo Cube Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            val = calibrate_from_frame(frame, min_area=args.min_area)
            if val is not None:
                pixels_per_cm = val

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

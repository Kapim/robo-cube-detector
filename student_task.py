#!/usr/bin/env python3
import time

from robot_vision_client import VisionClient



def main():
    vision = VisionClient()

    print("Kalibrace markeru...")
    try:
        vision.kalibruj_marker()
    except Exception as e:
        print(f"Kalibrace markeru selhala: {e}")

    while True:
        try:
            kostky = vision.detekuj_kostky()
        except Exception as e:
            print(f"Chyba komunikace se serverem: {e}")
            time.sleep(0.5)
            continue

        for k in kostky:
            if k.x_cm is None or k.y_cm is None:
                continue
            print(k)

        time.sleep(0.3)


if __name__ == "__main__":
    main()

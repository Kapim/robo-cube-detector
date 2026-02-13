#!/usr/bin/env python3
from dataclasses import dataclass
from typing import List, Optional

import requests


@dataclass
class Kostka:
    barva: str
    x_cm: Optional[float]
    y_cm: Optional[float]
    side_cm: Optional[float]


class VisionClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8765", timeout_s: float = 2.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _get(self, path: str):
        resp = requests.get(f"{self.base_url}{path}", timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str):
        resp = requests.post(f"{self.base_url}{path}", timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()

    def stav(self):
        return self._get("/status")

    def detekuj_kostky(self) -> List[Kostka]:
        data = self._get("/detect_cubes")
        out = []
        for c in data.get("cubes", []):
            out.append(
                Kostka(
                    barva=c.get("color"),
                    x_cm=c.get("robot_x_cm"),
                    y_cm=c.get("robot_y_cm"),
                    side_cm=c.get("side_cm"),
                )
            )
        return out

    def kalibruj_marker(self) -> bool:
        data = self._post("/calibrate_marker")
        return bool(data.get("ok", False))

    def kalibruj_px(self) -> bool:
        data = self._post("/calibrate_px")
        return bool(data.get("ok", False))

    def vypni_server(self) -> bool:
        data = self._post("/shutdown")
        return bool(data.get("ok", False))

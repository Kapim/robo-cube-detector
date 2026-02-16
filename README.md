# Robo Cube Detector

Projekt ma 2 casti:
- `vision_server.py`: server pro detekci kostek + kalibraci (bezi na stejnem PC)
- `robot_vision_client.py`: jednoduche API pro studentske skripty

Aplikace detekuje barevne kostky:
- zelena
- cervena
- zluta
- modra

## Instalace

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Spusteni serveru (ucitel)

```bash
python vision_server.py \
  --show-window \
  --camera 0 \
  --width 1280 \
  --height 720 \
  --marker-id 23 \
  --marker-size-cm 5.0 \
  --marker-origin-robot-x 12.3 \
  --marker-origin-robot-y 0.0 \
  --marker-to-robot-yaw-deg 0 \
  --size-max-scale 1.80
```

HTTP endpointy bezici na localhost:
- `GET /health`
- `GET /status`
- `GET /detect_cubes`
- `POST /calibrate_marker`
- `POST /calibrate_px`
- `POST /shutdown`

OpenCV ovladani v okne:
- `m` = rucni kalibrace markeru
- `c` = fallback kalibrace px/cm
- `q` = konec serveru

## 2) Studentske API (student)

Minimalni pouziti:

```python
from robot_vision_client import VisionClient

vision = VisionClient()  # localhost:8765
kostky = vision.detekuj_kostky()
for k in kostky:
    print(k.barva, k.x_cm, k.y_cm)
```

Dostupne metody:
- `detekuj_kostky()` -> `list[Kostka]`
- `kalibruj_marker()` -> `bool`
- `kalibruj_px()` -> `bool`
- `stav()` -> `dict`
- `vypni_server()` -> `bool`

## Ukazkovy studentsky skript

```bash
python student_task.py
```

Soubor `student_task.py` ukazuje jednoduche detekovaní kostek.

## Poznamky ke kalibraci

Doporuceny postup:
1. Vytiskni marker a dej ho na stul do stale pozice.
2. Zmer jednou transformaci marker -> robot (X, Y, yaw).
   Pokud je marker primo pred robotem bez natoceni, pouzij `--marker-to-robot-yaw-deg 0`.
3. Spust `vision_server.py`.
4. Stiskni `m` (nebo zavolej `POST /calibrate_marker`).

Po kalibraci se drzi posledni ulozena transformace, takze kratkodobe zakryti markeru nevadi (pokud se kamera/robot/stul nepohnou).

# Robo Cube Detector

Jednoducha aplikace v Pythonu, ktera pres OpenCV detekuje na stole barevne kosticky:
- zelena
- cervena
- zluta
- modra

Aplikace umi po kalibraci filtrovat objekty podle cilove hrany `2.5 cm`.

## Instalace

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Spusteni

```bash
python app.py
```

Volitelne parametry:

```bash
python app.py --camera 0 --width 1280 --height 720 --min-area 450
```

## Ovládání

- `c` = kalibrace velikosti (dej do zaberu jednu kosticku s hranou 2.5 cm)
- `q` = konec

Po kalibraci aplikace zobrazuje odhad velikosti v cm a ponechava jen objekty blizke hranici 2.5 cm.

## Poznamky

- Pro stabilni vysledky pouzij rovnomerne osvetleni.
- Pokud kamera meni expozici, pomuze fixni osvetleni stolu.
- HSV rozsahy barev lze doladit primo v `app.py`.

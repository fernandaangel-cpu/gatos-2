# Tarot Cam Meme Detector

Point your webcam at yourself, make a gesture, and get the matching Tarot card meme in real time. Runs either as a desktop app (OpenCV windows) or entirely in the browser (MediaPipe WASM, no install).

Two windows/panes side by side: 
- **Camera** — your webcam feed with hand landmarks drawn on top, plus a live debug readout in the corner
- **Meme** — the Tarot card matching whatever gesture you're currently making

## Jerarquía de Gestos (cartas)

Checked in this order — when a pose matches, earlier ones win:

| # | Carta | como se activa |imagen |
|---|---|---|---|
| 1 | **Muerte** | Cabeza inclinada lateralmente $\ge 20^\circ$, con una oreja acercándose al hombro | `muerte.png` |
| 2 | **Sol** | Ambos brazos levantados, con las manos sobre la cabeza y las puntas de los dedos enfrentadas formando un arco | `sol.jpeg` |
| 3 | **Mago** | Un brazo extendido verticalmente hacia arriba, mano sobre la cabeza | `Mago.jpeg` |
| 4 | **Amantes** | Ambas manos frente al pecho, índices y pulgares unidos formando un corazón | `Amantes.jpeg` |
| 5 | **Diablo** | Ambas manos junto a la parte superior de la cabeza, ambos índices extendidos hacia arriba | `Diablo.jpeg` |
| 6 | **El Loco** | Ambos índices extendidos apuntando hacia las sienes | `Elloco.jpeg` |
| 7 | **Emperador** | Cabeza frontal, manos abajo y brazos relajados (Default) | `emperador.jpeg` |

- [carpeta de imágenes](cartas)

- [video](video)
  
Meme images live in `cartas/`.

## Running it — desktop (Python)

Requires Python 3 and a webcam.

Easiest way: just double-click **`Launch Gesture Meme.command`**. First run takes a minute to set itself up (installs everything automatically), then launches straight away.

Or manually in Terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 gesture_meme.py
```

Press `q` or `Esc` in the Camera window to quit.

## Running it — browser

No install needed:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000` in Google Chrome and allow camera access.

## Project layout

```
gesture_meme.py   desktop version (OpenCV + MediaPipe Python tasks API)
app.js            browser version (MediaPipe tasks-vision WASM)
index.html        browser UI shell
cartas/           Tarot card images
memes/            Legacy meme images and cartas
models/           MediaPipe .task model files used by the desktop version
requirements.txt  Python dependencies
```

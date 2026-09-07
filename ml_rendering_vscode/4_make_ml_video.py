from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent
FRAME_DIR = ROOT / 'renders' / 'ml_360'
OUTPUT = ROOT / 'ml_nerf_360.mp4'
FPS = 24

frames = sorted(FRAME_DIR.glob('frame_*.png'))
if not frames:
    raise SystemExit(f'No frames found in {FRAME_DIR}')

first = cv2.imread(str(frames[0]))
h, w = first.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(str(OUTPUT), fourcc, FPS, (w, h))

for path in frames:
    img = cv2.imread(str(path))
    if img is None:
        continue
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h))
    writer.write(img)

writer.release()
print('Created:', OUTPUT)

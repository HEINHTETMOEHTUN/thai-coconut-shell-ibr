from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from importlib import util

ROOT = Path(__file__).resolve().parent
DATASET_JSON = ROOT / 'dataset' / 'dataset.json'
CHECKPOINT = ROOT / 'checkpoints' / 'latest.pt'

# Import model helpers from 2_train_nerf.py without renaming the file.
spec = util.spec_from_file_location('train_mod', ROOT / '2_train_nerf.py')
train_mod = util.module_from_spec(spec)
spec.loader.exec_module(train_mod)
TinyNeRF = train_mod.TinyNeRF
volume_render = train_mod.volume_render
rays_for_pixels = train_mod.rays_for_pixels


def look_at_pose(theta_deg: float, elevation_deg: float, radius: float) -> np.ndarray:
    theta = math.radians(theta_deg)
    phi = math.radians(elevation_deg)
    pos = np.array([
        radius * math.sin(theta) * math.cos(phi),
        radius * math.sin(phi),
        radius * math.cos(theta) * math.cos(phi),
    ], dtype=np.float32)
    forward = -pos
    forward /= np.linalg.norm(forward) + 1e-8
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right) + 1e-8
    up = np.cross(right, forward)
    up /= np.linalg.norm(up) + 1e-8
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = pos
    return c2w


def render_image(model, device, pose_np, fx, fy, cx, cy, size, chunk=2048):
    ys, xs = np.mgrid[0:size, 0:size]
    xs = torch.from_numpy(xs.reshape(-1).astype(np.float32)).to(device)
    ys = torch.from_numpy(ys.reshape(-1).astype(np.float32)).to(device)
    pose = torch.tensor(pose_np, dtype=torch.float32, device=device)
    result = []
    with torch.no_grad():
        for start in range(0, xs.numel(), chunk):
            x = xs[start:start+chunk]
            y = ys[start:start+chunk]
            ro, rd = rays_for_pixels(
                pose,
                torch.tensor(fx, device=device),
                torch.tensor(fy, device=device),
                torch.tensor(cx, device=device),
                torch.tensor(cy, device=device),
                x,
                y,
            )
            rgb, _ = volume_render(model, ro, rd, perturb=False)
            result.append(rgb.detach().cpu())
    rgb = torch.cat(result, dim=0).numpy().reshape(size, size, 3)
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=120)
    parser.add_argument('--elevation', type=float, default=10.0)
    parser.add_argument('--output', type=str, default='renders/ml_360')
    parser.add_argument('--chunk', type=int, default=2048)
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    with open(DATASET_JSON) as f:
        meta = json.load(f)
    size = int(meta['image_size'])
    radius = float(meta['camera_radius'])

    # Use median intrinsics from all processed frames.
    fxs = [fr['fx'] for fr in meta['frames']]
    fys = [fr['fy'] for fr in meta['frames']]
    fx = float(np.median(fxs))
    fy = float(np.median(fys))
    cx = size / 2.0
    cy = size / 2.0

    model = TinyNeRF().to(device)
    ckpt = torch.load(CHECKPOINT, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Rendering on', device)
    print('Frames:', args.frames)
    print('Elevation:', args.elevation)
    for i in tqdm(range(args.frames)):
        theta = -360.0 * i / args.frames
        pose = look_at_pose(theta, args.elevation, radius)
        frame = render_image(model, device, pose, fx, fy, cx, cy, size, chunk=args.chunk)
        cv2.imwrite(str(out_dir / f'frame_{i:04d}.png'), frame)

    print('Frames saved to:', out_dir)


if __name__ == '__main__':
    main()

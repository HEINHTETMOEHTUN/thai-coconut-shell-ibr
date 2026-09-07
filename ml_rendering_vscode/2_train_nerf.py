from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

ROOT = Path(__file__).resolve().parent
DATASET_JSON = ROOT / 'dataset' / 'dataset.json'
CHECKPOINT_DIR = ROOT / 'checkpoints'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

NEAR = 1.5
FAR = 4.5
N_SAMPLES = 48
WHITE_BACKGROUND = True


def positional_encoding(x, levels):
    out = [x]
    for i in range(levels):
        freq = (2.0 ** i) * math.pi
        out.append(torch.sin(freq * x))
        out.append(torch.cos(freq * x))
    return torch.cat(out, dim=-1)


class TinyNeRF(nn.Module):
    def __init__(self, hidden=128, xyz_levels=8, dir_levels=4):
        super().__init__()
        self.xyz_levels = xyz_levels
        self.dir_levels = dir_levels
        xyz_dim = 3 * (1 + 2 * xyz_levels)
        dir_dim = 3 * (1 + 2 * dir_levels)

        self.fc1 = nn.Linear(xyz_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)
        self.fc5 = nn.Linear(hidden + xyz_dim, hidden)
        self.fc6 = nn.Linear(hidden, hidden)
        self.fc7 = nn.Linear(hidden, hidden)
        self.fc8 = nn.Linear(hidden, hidden)

        self.sigma = nn.Linear(hidden, 1)
        self.feature = nn.Linear(hidden, hidden)
        self.color1 = nn.Linear(hidden + dir_dim, hidden // 2)
        self.color2 = nn.Linear(hidden // 2, 3)

    def forward(self, xyz, dirs):
        xenc = positional_encoding(xyz, self.xyz_levels)
        denc = positional_encoding(dirs, self.dir_levels)

        h = F.relu(self.fc1(xenc))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc4(h))
        h = torch.cat([h, xenc], dim=-1)
        h = F.relu(self.fc5(h))
        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        h = F.relu(self.fc8(h))

        sigma = F.softplus(self.sigma(h))
        feat = self.feature(h)
        c = torch.cat([feat, denc], dim=-1)
        c = F.relu(self.color1(c))
        rgb = torch.sigmoid(self.color2(c))
        return rgb, sigma


def rays_for_pixels(pose, fx, fy, cx, cy, xs, ys):
    # Camera looks along -Z. Image Y increases downward, so camera-space Y is negated.
    dirs_cam = torch.stack([
        (xs - cx) / fx,
        -(ys - cy) / fy,
        -torch.ones_like(xs),
    ], dim=-1)
    dirs_cam = F.normalize(dirs_cam, dim=-1)

    R = pose[:3, :3]
    t = pose[:3, 3]
    dirs_world = dirs_cam @ R.T
    dirs_world = F.normalize(dirs_world, dim=-1)
    origins = t.expand_as(dirs_world)
    return origins, dirs_world


def volume_render(model, rays_o, rays_d, perturb=True):
    device = rays_o.device
    n_rays = rays_o.shape[0]
    t_vals = torch.linspace(NEAR, FAR, N_SAMPLES, device=device)
    t_vals = t_vals.expand(n_rays, N_SAMPLES)

    if perturb:
        mids = 0.5 * (t_vals[:, :-1] + t_vals[:, 1:])
        upper = torch.cat([mids, t_vals[:, -1:]], dim=-1)
        lower = torch.cat([t_vals[:, :1], mids], dim=-1)
        t_rand = torch.rand_like(t_vals)
        t_vals = lower + (upper - lower) * t_rand

    pts = rays_o[:, None, :] + rays_d[:, None, :] * t_vals[..., None]
    dirs = rays_d[:, None, :].expand_as(pts)

    flat_pts = pts.reshape(-1, 3)
    flat_dirs = dirs.reshape(-1, 3)
    rgb, sigma = model(flat_pts, flat_dirs)
    rgb = rgb.reshape(n_rays, N_SAMPLES, 3)
    sigma = sigma.reshape(n_rays, N_SAMPLES)

    deltas = t_vals[:, 1:] - t_vals[:, :-1]
    deltas = torch.cat([deltas, torch.full_like(deltas[:, :1], 1e10)], dim=-1)
    alpha = 1.0 - torch.exp(-sigma * deltas)
    trans = torch.cumprod(torch.cat([torch.ones((n_rays, 1), device=device), 1.0 - alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
    weights = alpha * trans

    rgb_map = torch.sum(weights[..., None] * rgb, dim=1)
    acc_map = torch.sum(weights, dim=1)
    if WHITE_BACKGROUND:
        rgb_map = rgb_map + (1.0 - acc_map)[..., None]
    return rgb_map, acc_map


class Dataset:
    def __init__(self, metadata_path: Path):
        with open(metadata_path) as f:
            meta = json.load(f)
        self.root = metadata_path.parent
        self.size = int(meta['image_size'])
        self.frames = meta['frames']
        self.images = []
        self.masks = []
        for fr in self.frames:
            img = cv2.imread(str(self.root / fr['rgb']))
            mask = cv2.imread(str(self.root / fr['mask']), cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                raise RuntimeError(f"Missing processed image for {fr['source_name']}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.images.append(img)
            self.masks.append(mask)

    def sample(self, batch_rays, device):
        idx = random.randrange(len(self.frames))
        fr = self.frames[idx]
        img = self.images[idx]
        mask = self.masks[idx]
        h, w = mask.shape

        fg_y, fg_x = np.where(mask > 127)
        bg_y, bg_x = np.where(mask <= 127)
        n_fg = int(batch_rays * 0.75)
        n_bg = batch_rays - n_fg

        if len(fg_x) == 0:
            n_fg = 0
            n_bg = batch_rays
        if len(bg_x) == 0:
            n_bg = 0
            n_fg = batch_rays

        xs_list, ys_list = [], []
        if n_fg > 0:
            sel = np.random.randint(0, len(fg_x), size=n_fg)
            xs_list.append(fg_x[sel])
            ys_list.append(fg_y[sel])
        if n_bg > 0:
            sel = np.random.randint(0, len(bg_x), size=n_bg)
            xs_list.append(bg_x[sel])
            ys_list.append(bg_y[sel])

        xs = np.concatenate(xs_list).astype(np.float32)
        ys = np.concatenate(ys_list).astype(np.float32)
        order = np.random.permutation(len(xs))
        xs, ys = xs[order], ys[order]

        target_rgb = img[ys.astype(np.int64), xs.astype(np.int64)].astype(np.float32) / 255.0
        target_mask = (mask[ys.astype(np.int64), xs.astype(np.int64)] > 127).astype(np.float32)

        xs_t = torch.from_numpy(xs).to(device)
        ys_t = torch.from_numpy(ys).to(device)
        pose = torch.tensor(fr['transform_matrix'], dtype=torch.float32, device=device)
        rays_o, rays_d = rays_for_pixels(
            pose,
            torch.tensor(fr['fx'], device=device),
            torch.tensor(fr['fy'], device=device),
            torch.tensor(fr['cx'], device=device),
            torch.tensor(fr['cy'], device=device),
            xs_t,
            ys_t,
        )
        return (
            rays_o,
            rays_d,
            torch.from_numpy(target_rgb).to(device),
            torch.from_numpy(target_mask).to(device),
            fr,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=12000)
    parser.add_argument('--batch-rays', type=int, default=768)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--save-every', type=int, default=1000)
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('=' * 60)
    print(' TINY NeRF TRAINING')
    print('=' * 60)
    print('Device:', device)
    print('Steps:', args.steps)
    print('Batch rays:', args.batch_rays)

    dataset = Dataset(DATASET_JSON)
    print('Training frames:', len(dataset.frames))

    model = TinyNeRF().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start = time.time()
    ema_loss = None
    pbar = trange(1, args.steps + 1)
    for step in pbar:
        rays_o, rays_d, target_rgb, target_mask, fr = dataset.sample(args.batch_rays, device)
        pred_rgb, pred_acc = volume_render(model, rays_o, rays_d, perturb=True)

        rgb_loss = F.mse_loss(pred_rgb, target_rgb)
        mask_loss = F.mse_loss(pred_acc, target_mask)
        loss = rgb_loss + 0.10 * mask_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        value = float(loss.detach().cpu())
        ema_loss = value if ema_loss is None else 0.95 * ema_loss + 0.05 * value
        if step % 20 == 0:
            psnr = -10.0 * math.log10(max(float(rgb_loss.detach().cpu()), 1e-10))
            pbar.set_description(f'loss={ema_loss:.5f} psnr={psnr:.2f}dB ring={fr["ring"]}')

        if step % args.save_every == 0 or step == args.steps:
            ckpt = {
                'step': step,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'config': {
                    'near': NEAR,
                    'far': FAR,
                    'n_samples': N_SAMPLES,
                    'white_background': WHITE_BACKGROUND,
                },
            }
            path = CHECKPOINT_DIR / f'nerf_step_{step:06d}.pt'
            torch.save(ckpt, path)
            torch.save(ckpt, CHECKPOINT_DIR / 'latest.pt')

    elapsed = time.time() - start
    print('\nTraining complete.')
    print(f'Time: {elapsed/60:.1f} minutes')
    print('Checkpoint:', CHECKPOINT_DIR / 'latest.pt')


if __name__ == '__main__':
    main()

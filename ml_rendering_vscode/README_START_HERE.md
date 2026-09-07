# ML Rendering in VS Code — Thai Coconut-Shell Owl

This starter uses a small NeRF implemented directly in PyTorch so it can use Apple Silicon MPS from VS Code without NVIDIA CUDA.

## Expected project structure

```text
thai-coconut-shell-ibr/
├── images/
│   ├── horizontal/   # 36 images, 10° steps
│   ├── upper/        # 18 images, 20° steps, ~25° elevation
│   └── upper45/      # 18 images, 20° steps, ~45° elevation
├── output/
└── ml_rendering/
    ├── 0_check_mps.py
    ├── 1_prepare_ml.py
    ├── 2_train_nerf.py
    ├── 3_render_nerf.py
    ├── 4_make_ml_video.py
    └── requirements.txt
```

## Run order

1. Create/activate `.venv`
2. `pip install -r requirements.txt`
3. `python 0_check_mps.py`
4. `python 1_prepare_ml.py`
5. VISUALLY inspect `dataset/rgb` and `dataset/masks`
6. Quick test: `python 2_train_nerf.py --steps 2000 --batch-rays 512`
7. Render quick test: `python 3_render_nerf.py --frames 36 --elevation 10`
8. If good, full training: `python 2_train_nerf.py --steps 12000 --batch-rays 768`
9. Final render: `python 3_render_nerf.py --frames 120 --elevation 10`
10. `python 4_make_ml_video.py`

## Important tuning

- In `1_prepare_ml.py`, `UPPER` is assumed to be ~25° elevation and `upper45` is 45°.
- `ANGLE_SIGN=-1` assumes increasing photo index corresponds to a clockwise object rotation. If the learned geometry appears mirrored or validation is reversed, set it to `+1`, rerun preparation, and retrain.
- Do not train until the masks look good.

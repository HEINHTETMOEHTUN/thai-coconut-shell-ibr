# Thai Coconut-Shell Project — Research Evaluation

## Experiment 1: Rendering hold-out + ablation

Place `9_rendering_evaluation.py` inside the project's `output/` folder.

Run:

```bash
cd /Users/hyan/Desktop/thai-coconut-shell-ibr/output
python3 9_rendering_evaluation.py
```

The script uses a real held-out photograph as ground truth:

- View 01 + View 03 -> predict View 02
- View 03 + View 05 -> predict View 04
- ...

It compares:

A. Baseline: one-way optical flow + linear blending
B. + Lighting normalization
C. + Bidirectional flow + consistency/occlusion/confidence
D. Full proposed: C + Laplacian pyramid blending

Outputs are written under:

`../evaluation/rendering/`

Important files:

- `summary_results.csv`
- `per_case_results.csv`
- `comparisons/`
- `charts/` (created automatically if matplotlib is installed)

Metrics:

- MAE: lower is better
- PSNR: higher is better
- SSIM: higher is better
- Edge F1: higher is better
- Runtime: lower is faster

Optional chart dependency:

```bash
python3 -m pip install matplotlib
```

## Experiment 2: 3D reconstruction comparison

Place `10_reconstruction_compare.swift` in `output/`.

It compares:

- Horizontal ring only
- Horizontal + upper photographs

Both models use the same RealityKit settings and `.medium` detail for a fair comparison.

Compile:

```bash
xcrun swiftc -parse-as-library 10_reconstruction_compare.swift -o reconstruction_compare -framework RealityKit
```

Run:

```bash
./reconstruction_compare
```

Outputs are written under:

`../evaluation/reconstruction/`

Important files:

- `horizontal_only.usdz`
- `horizontal_plus_upper.usdz`
- `reconstruction_results.csv`

## Visual 3D completeness scoring

Use `reconstruction_visual_evaluation_template.csv`.

Open both models from exactly five directions:

1. Front
2. Right
3. Back
4. Left
5. Top

For each direction use:

- 0 = missing / severely broken geometry
- 1 = partially correct / visible artifact
- 2 = complete and structurally correct

Then:

`Geometry Total = Front + Right + Back + Left + Top`

`Geometry Completeness (%) = Geometry Total / 10 * 100`

Also score texture consistency:

- 0 = poor / badly stretched
- 1 = moderate artifacts
- 2 = clean and consistent

Do not call this "3D accuracy" because there is no laser-scan ground truth. Report it as a standardized visual completeness evaluation.

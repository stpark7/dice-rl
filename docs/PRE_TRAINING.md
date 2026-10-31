## Hyperparameters

| Hyperparameter | Click Mouse | Fold Glasses | Hammer Nail | Pick Bucket | Pinch Tongs | Water Plant |
|---|---|---|---|---|---|---|
| Proprioceptive dim | 23 | 23 | 23 | 23 | 23 | 23 |
| Action dim | 22 | 22 | 22 | 22 | 22 | 22 |
| Cameras | front, wrist | front, wrist | front, wrist | front, wrist | front, wrist | front, wrist |
| Image resolution | 96 × 96 | 96 × 96 | 96 × 96 | 96 × 96 | 96 × 96 | 96 × 96 |
| Crop size | 84 × 84 | 84 × 84 | 84 × 84 | 84 × 84 | 84 × 84 | 84 × 84 |
| U-Net base channels | 128 | 128 | 128 | 128 | 128 | 128 |
| U-Net channels | [128, 256, 256] | [128, 256, 256] | [128, 256, 256] | [128, 256, 256] | [128, 256, 256] | [128, 256, 256] |
| Flow steps | 10 | 10 | 10 | 10 | 10 | 10 |
| Prediction / execution steps | 16 / 16 | 16 / 16 | 16 / 16 | 16 / 16 | 16 / 16 | 16 / 16 |
| Used demos (50 / all) | 99 | 100 | 99 | 98 | 100 | 100 |
| Max episode steps | 500 | 800 | 400 | 700 | 700 | 400 |

Shared settings:

- **Model:** ResNet-18 encoders trained from scratch; one observation frame; Euler flow integration.
- **Training:** 2,000 epochs, batch size 256, seed 42. AdamW with LR `1e-4` and weight decay `1e-6`; cosine LR schedule with 100 warmup epochs and minimum LR `1e-5`.


## BC Policy Success Rate

Success rates are measured over 200 evaluation episodes per checkpoint.

| Task | Selected checkpoint | Success rate (%) | Mean episode length | Success episode length |
|---|---|---:|---:|---:|
| Click Mouse | | | | |
| Fold Glasses | `state_1100.pt` | 59 | | |
| Hammer Nail | `state_1500.pt` | 50 | | |
| Pick Bucket | `state_2000.pt` | 78 | 465.6 | 399.5 |
| Pinch Tongs | `state_700.pt` | 49 | | |
| Water Plant | | | | |

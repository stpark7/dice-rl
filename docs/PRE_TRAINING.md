## Hyperparameters

| Hyperparameter | Click Mouse | Fold Glasses | Pick Bucket | Pinch Tongs | Water Plant |
|---|---|---|---|---|---|
| Proprioceptive dim | 23 | 23 | 23 | 23 | 23 |
| Action dim | 22 | 22 | 22 | 22 | 22 |
| Cameras | front, wrist | front, wrist | front, wrist | front, wrist | front, wrist |
| Image resolution | 96 × 96 | 96 × 96 | 96 × 96 | 96 × 96 | 96 × 96 |
| Crop size | 84 × 84 | 84 × 84 | 84 × 84 | 84 × 84 | 84 × 84 |
| U-Net base channels | 128 | 256 | 128 | 128 | 128 |
| U-Net channels | [128, 256, 256] | [256, 512, 512] | [128, 256, 256] | [128, 256, 256] | [128, 256, 256] |
| Flow steps | 10 | 10 | 10 | 10 | 10 |
| Prediction / execution steps | 16 / 16 | 16 / 16 | 16 / 16 | 16 / 16 | 16 / 16 |
| Used demos (all) | 99 | 100 | 98 | 100 | 100 |
| Training epochs | 3,000 | 4,000 | 2,000 | 3,000 | 2,000 |
| LR cycle (epochs) | 3,000 | 5,000 | 2,000 | 3,000 | 2,000 |
| Max episode steps | 500 | 800 | 700 | 700 | 400 |

Shared settings:

- **Model:** ResNet-18 encoders trained from scratch; one observation frame; Euler flow integration.
- **Training:** Batch size 256, seed 42. AdamW with LR `1e-4` and weight decay `1e-6`; cosine LR schedule with 100 warmup epochs and minimum LR `1e-5`.

## BC Policy Success Rate

| Task | Selected checkpoint | Hugging Face | Success rate (%) |
|---|---|---|---:|
| Click Mouse | `state_2000.pt` | [Checkpoints](https://huggingface.co/robopark/dexjoco-pretrian-checkpoints/tree/main/pretrained_bc_policy_click_mouse_img) | 80 |
| Fold Glasses | `state_2000.pt` | [Checkpoints](https://huggingface.co/robopark/dexjoco-pretrian-checkpoints/tree/main/pretrained_bc_policy_fold_glasses_img) | 41 |
| Pick Bucket | `state_2000.pt` | [Checkpoints](https://huggingface.co/robopark/dexjoco-pretrian-checkpoints/tree/main/pretrained_bc_policy_pick_bucket_img) | 78 |
| Pinch Tongs | `state_2000.pt` | [Checkpoints](https://huggingface.co/robopark/dexjoco-pretrian-checkpoints/tree/main/pretrained_bc_policy_pinch_tongs_img) | 41 |
| Water Plant | `state_2000.pt` | [Checkpoints](https://huggingface.co/robopark/dexjoco-pretrian-checkpoints/tree/main/pretrained_bc_policy_water_plant_img) | 85 |


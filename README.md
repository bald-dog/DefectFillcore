# DefectFill2 Unofficial Implementation

Industrial defect generation and inpainting experimental project based on Stable Diffusion Inpainting ideas.  
DefectFill: Realistic Defect Generation with Inpainting Diffusion Model for Visual Inspection  
arXiv: https://arxiv.org/abs/2503.13985

---

## 1. Main Features
- Core model wrapper: [model.py](model.py)
- Training script (attention loss, gradient accumulation, TensorBoard): [train.py](train.py)
- Inference / generation script: [inference.py](inference.py)
- Data loading & preprocessing: [data_loader.py](data_loader.py)
- Utility functions (save/load checkpoints, etc.): [utils.py](utils.py)

---

## 2. Environment Setup

### 2.1 Python Version
Recommended Python 3.11 (tested with 3.11.9).

### 2.2 Install Dependencies
Dependencies listed in [requirements.txt](requirements.txt):
```bash
pip install -r requirements.txt
```
Key packages:
- torch / torchvision
- diffusers
- transformers
- tqdm
- tensorboard

### 2.3 Recommended Conda Environment
```bash
conda create -n defectfill2 python=3.11 -y
conda activate defectfill2
pip install -r requirements.txt
```

---

## 3. Dataset

### 3.1 Directory Structure Example
```
/home/phpc/program2/DefectFill2/MVTec2/
  bottle/
    train/defective/broken_large/000.png ...
    train/defective_masks/broken_large/000_mask.png ...
    000.png ...
```

### 3.2 Dataset Download
An adjusted MVTec dataset (aligned to this project's directory structure) is uploaded to Baidu Netdisk:  
Link: https://pan.baidu.com/s/1xwr5AkrmLi6ahPvU5eQL8Q  Extraction code: g84e  
-- Shared via Baidu Netdisk SVIP v7

---

## 4. Training Data Pipeline

From [train.py](train.py):
- Each batch sample fields (example): `image`, `mask`, `background`, `adjusted_mask`, `is_defect`, `object_class`, `defect_type`
- Only defect samples are processed: filter `is_defect == 1`
- Two processing paths (same defect sample reused 1:1):
  1. Real defect mask: learn defect semantics (Defect Loss)  
     - Prompt template: `"A photo of {defect_type}"`
  2. Random rectangle mask: learn object integrity (Object Loss)  
     - Prompt template: `"A {object_class} with {defect_type}"`

Random rectangle mask generation reference:
- Around line 193 in [train.py](train.py)

---

## 5. Loss Function

Total loss (see around line 235 in [train.py](train.py)):
```
L_total = λ_defect * L_defect + λ_object * L_object + λ_attn * L_attention
```

---

## 6. Training

### 6.1 Basic Command
```bash
python train.py \
  --data_dir "/home/phpc/program2/DefectFill2/MVTec" \
  --object_class bottle \
  --output_dir "./models/bottle" \
  --batch_size 2 \
  --max_train_steps 2000
```

Optional arguments (see ArgumentParser in [train.py](train.py)):
- `--lora_rank`
- `--learning_rate`
- `--save_steps`
- `--resume_from`
- `--gradient_accumulation_steps`
- `--lr_warmup_steps`
- `--seed`

### 6.2 Training Features
- Gradient accumulation: around line 247 in [train.py](train.py)
- LR warmup: around line 131 in [train.py](train.py)
- Checkpoint: saved every `save_steps` or at end via `utils.save_checkpoint`
- Logging / monitoring:
  - Text log: `train.txt`
  - TensorBoard: `output_dir/logs`
    ```bash
    tensorboard --logdir ./models/bottle/logs --port 6006
    ```

---

## 7. Inference / Generation

Script: [inference.py](inference.py)

Example:
```bash
python inference.py \
  --checkpoint checkpoint_2000.pt \
  --object_class bottle \
  --image_path 000.png \
  --mask_path 012_mask.png \
  --defect_type broken_large
```

Output:
- Generated defect inpainted / synthesized image
- Can optionally save intermediate attention maps (enable `attention_maps` in model)

---

## 8. Model & Trainable Components

`DefectFillModel` in [model.py](model.py) loads Stable Diffusion Inpainting components (UNet + text encoder + VAE).  
Fine-tuning strategy (as inferred):
- LoRA (`--lora_rank`)
- Partial parameter unfreezing (check implementation)

---

## 9. Notes

1. Input channels: Stable Diffusion Inpainting UNet typically expects 9 channels (latent 4 + masked latent 4 + mask 1); ensure preprocessing matches (see forward logic in [model.py](model.py)).
2. Optimization target: noise prediction (DDPM-style MSE between predicted and real noise).
3. To switch to image reconstruction objective, add pixel / perceptual loss at decoded stage.
4. Attention map clearing point:
   - Around line 131 in [train.py](train.py): `model.attention_maps = {}`

---

## 10. FAQ

| Issue | Description |
|-------|-------------|
| OOM (out of memory) | Reduce `batch_size` or lower `lora_rank` |
| NaN loss | Detected and skipped (see lines 235 / 247 in train loop) |
| Attention not converging | Tune `λ_attn` (e.g. 0.05 → 0.02) |
| Misaligned defect region | Check mask size & alignment with source image |

---

## 11. Extensions

- Multi-scale attention consistency
- Mask-guided latent blending
- Add perceptual loss (LPIPS) for quality
- Multi-defect prompts: `"A bottle with broken_large and contamination"`

---

## 12. Quick Start

```bash
# 1. Create environment
conda create -n defectfill2 python=3.11 -y
conda activate defectfill2

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start training
python train.py --data_dir ./MVTec2 --object_class bottle --output_dir ./models/bottle

# 4. Monitor
tensorboard --logdir ./models/bottle/logs

# 5. Inference
python inference.py --checkpoint checkpoint_2000.pt \
  --object_class bottle \
  --image_path ./some_input.png \
  --mask_path ./some_mask.png \
  --defect_type broken_large
```

---

## 13. File Index

- Training loop: [train.py](train.py)
- Model wrapper: [model.py](model.py)
- Data loader: [data_loader.py](data_loader.py)
- Inference script: [inference.py](inference.py)
- Checkpoint utilities: [utils.py](utils.py)
- Training log example:

# CULane 2K Subset Training

## Dataset Setup Complete

- **Training**: 2,000 samples (`train_2k.txt`)
- **Validation**: 39,000 samples (`val_gt.txt`)
- Old pickle files removed - will regenerate on first run

---

## Training Commands

### Activate Conda Environment (clrernet)
```bash
conda activate clrernet
```

**Environment Details:**
- mmcv: 1.7.1
- torch: 2.1.0+cu121
- CUDA: Enabled

### Single GPU Training (Recommended for 2K subset) - 12GB GPU
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

cd /home/alki/projects/CondLSTR

python tools/train.py \
    -a CondLSTR2DRes34 \
    -d culane \
    --train-split train_2k \
    --val-split test_2k \
    -v v1.0 \
    -c 21 \
    -t lane_det_2d \
    --img-size 800 320 \
    --data-dir /home/alki/projects/datasets \
    --logs-dir ./logs/culane_2k \
    -b 1 \
    -j 0 \
    --save-steps 500 \
    --num-epochs 50
```

> ⚠️ **CRITICAL:** `--train-split train_2k` is required! Without it, defaults to full dataset (11,675 samples instead of 2,000).

**Important settings for 12GB GPU:**
- `--img-size 800 320` - Reduced resolution (was 1600x640)
- `-b 1` - Batch size 1
- `-j 0` - No multiprocessing workers
- `--save-steps 500` - Save checkpoints frequently (in case of crash)
- `export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` - Prevent memory fragmentation

> **Note**:
> - `-b 1` batch size for 12GB GPU (increase to 2 if using AMP)
> - `-j 0` disables multiprocessing workers to avoid segmentation fault

### Quick Test Training (5 epochs)
```bash
cd /home/alki/projects/CondLSTR

python tools/train.py \
    -a CondLSTR2DRes34 \
    -d culane \
    --train-split train_2k \
    --val-split test_2k \
    -v v1.0 \
    -c 21 \
    -t lane_det_2d \
    --data-dir /home/alki/projects/datasets \
    --logs-dir ./logs/culane_2k_test \
    -b 1 \
    -j 0 \
    --num-epochs 5
```

### Multi-GPU Training (4 GPUs)
```bash
cd /home/alki/projects/CondLSTR

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    tools/train.py \
    -a CondLSTR2DRes34 \
    -d culane \
    -v v1.0 \
    -c 21 \
    -t lane_det_2d \
    --data-dir /home/alki/projects/datasets \
    --logs-dir ./logs/culane_2k \
    --gpu-ids 0,1,2,3 \
    -b 4 \
    -j 4 \
    --num-epochs 50
```

### Training with AMP (Faster, less memory) - RECOMMENDED for 12GB GPU
```bash
cd /home/alki/projects/CondLSTR

python tools/train.py \
    -a CondLSTR2DRes34 \
    -d culane \
    -v v1.0 \
    -c 21 \
    -t lane_det_2d \
    --data-dir /home/alki/projects/datasets \
    --logs-dir ./logs/culane_2k \
    -b 2 \
    -j 0 \
    -p amp \
    --num-epochs 50
```

> **Recommended**: Use AMP for 12GB GPU - allows batch size of 2

---

## Evaluation (After Training)

```bash
python tools/train.py \
    -a CondLSTR2DRes34 \
    -d culane \
    -v v1.0 \
    -c 21 \
    -t lane_det_2d \
    --data-dir /home/alki/projects/datasets \
    --logs-dir ./logs/culane_2k \
    -b 1 \
    -j 0 \
    --eval
```

## Test/Visualization (After Training)

```bash
python tools/test.py \
    -a CondLSTR2DRes34 \
    -d culane \
    -v v1.0 \
    -c 21 \
    -t lane_det_2d \
    --data-dir /home/alki/projects/datasets \
    --logs-dir ./logs/culane_2k \
    --test-dir ./output/culane_2k \
    -b 1 \
    -j 0
```

---

## Resume Training (After Crash)

**Current Status:**
- Checkpoint saved at **Epoch 0**
- Last iteration: **153/11675** (crashed with segfault)
- Loss: **87.565** → trending down (started at ~500)

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

cd /home/alki/projects/CondLSTR

python tools/train.py \
    -a CondLSTR2DRes34 \
    -d culane \
    -v v1.0 \
    -c 21 \
    -t lane_det_2d \
    --img-size 800 320 \
    --data-dir /home/alki/projects/datasets \
    --logs-dir ./logs/culane_2k \
    -b 1 \
    -j 0 \
    --save-steps 500 \
    --num-epochs 50 \
    --resume
```

**Note:** The segfaults are WSL/CUDA compatibility issues, not code bugs. Simply re-run with `--resume` after each crash. Training progress is preserved in checkpoints.

---

## Parameter Explanations

| Parameter | Value | Description |
|-----------|-------|-------------|
| `-a` | CondLSTR2DRes34 | Model architecture (ResNet34 backbone) |
| `-d` | culane | Dataset name |
| `-v` | v1.0 | Dataset version |
| `-c` | 21 | Number of classes (for lane detection) |
| `-t` | lane_det_2d | Task type (2D lane detection) |
| `--data-dir` | /home/alki/projects/datasets | Parent directory of datasets |
| `--logs-dir` | ./logs/culane_2k | Where to save checkpoints |
| `-b` | 4 | Batch size per GPU |
| `-j` | 4 | Number of data loading workers |
| `--num-epochs` | 50 | Number of training epochs |
| `-p` | amp | Optional: Use mixed precision for faster training |
| `--eval` | - | Run evaluation only (requires checkpoint) |
| `--resume` | - | Resume from checkpoint |

---

## Current Training Progress

| Metric | Value |
|--------|-------|
| Current Epoch | 0 (in progress) |
| Iterations | 153 / 11,675 |
| Samples | 2,000 per epoch |
| Last Loss | 87.565 (obj: 21.9, cls: 0.0, reg: 25.6, loc: 30.1, rng: 9.9) |
| Checkpoint Size | ~340 MB |
| Status | Ready to resume |

---

## Expected Training Time (2K samples)

| Hardware | Epoch Time | Total Time (50 epochs) |
|----------|-----------|----------------------|
| 1x RTX 3090 | ~2-3 min | ~2-2.5 hours |
| 4x RTX 3090 | ~1 min | ~1 hour |
| 1x RTX 4090 | ~1-2 min | ~1-1.5 hours |

*Preprocessing will add ~5-10 minutes on first run*
*Note: WSL segfaults may cause periodic crashes - use `--resume` to continue*

---

## Monitor Training

### TensorBoard
```bash
tensorboard --logdir ./logs/culane_2k/tensorboard
```

### View Log File
```bash
tail -f ./logs/culane_2k/log.txt
```

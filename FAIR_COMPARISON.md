# Fair F1 Comparison: LSTR vs CondLSTR

## Critical Parameters for Fair Comparison

### ✅ Must Match for Fair F1 Comparison

| Parameter | LSTR Baseline | CondLSTR Current | Status |
|-----------|---------------|------------------|--------|
| **Dataset** | train_2k (2000) | train_2k (2000) | ✅ SAME |
| **Total Samples Seen** | 12,500 × 16 = 200,000 | Need to calculate | ⚠️ CHECK |
| **Learning Rate** | 0.0001 (1e-4) | 0.0002 (2e-4) | ❌ DIFFERENT |
| **Optimizer** | adam | AdamW | ❌ DIFFERENT |
| **Input Size** | 295 × 820 = 241,900 px | 800 × 320 = 256,000 px | ✅ SIMILAR |
| **Augmentation** | rand_color: true | Need to check | ⚠️ CHECK |

### ⚠️ Impacts Training Speed (Not F1)

| Parameter | LSTR | CondLSTR | Impact |
|-----------|------|----------|--------|
| Batch Size | 16 | 1 | Speed only (gradient accumulation compensates) |
| Workers | 5 | 0 | Speed only |
| GPU Count | Multi-GPU | 1 GPU | Speed only |

---

## 🔴 Critical Issues Found

### 1. **Total Training Samples Different**

**LSTR:**
- max_iter: 12,500 iterations
- batch_size: 16
- **Total samples seen: 12,500 × 16 = 200,000 samples**

**CondLSTR:**
- num_epochs: 50
- samples per epoch: 2,000
- batch_size: 1
- **Total samples seen: 50 × 2,000 × 1 = 100,000 samples**

❌ **CondLSTR sees only HALF the data!**

### 2. **Learning Rate Different**

- LSTR: `learning_rate: 0.0001` (1e-4)
- CondLSTR: `--lr 2e-4` (default in train.py line 48)

❌ **CondLSTR uses 2× higher learning rate**

### 3. **Optimizer Different**

- LSTR: `adam`
- CondLSTR: `AdamW` (includes weight decay)

❌ **Different optimization algorithms**

---

## ✅ Fix for Fair Comparison

### Option 1: Match Total Samples (Simple, No Gradient Accumulation)

```bash
# Calculate required epochs:
# LSTR: 200,000 samples total
# CondLSTR: 2,000 samples/epoch × 1 batch = 2,000 samples/epoch
# Required: 200,000 ÷ 2,000 = 100 epochs

conda activate clrernet
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

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
    --logs-dir ./logs/culane_2k_fair \
    -b 1 \
    -j 0 \
    --num-epochs 100 \
    --lr 0.0001 \
    --optim AdamW \
    -wd 0 \
    --save-steps 500
```

⚠️ **Note:** Without gradient accumulation, batch norm statistics will differ from LSTR's batch=16.

### Option 2: Match LSTR Exactly (Gradient Accumulation) ✅

Use gradient accumulation to simulate batch_size=16:

```bash
# Effective batch = 16, but process 1 at a time
# Accumulate gradients over 16 iterations before update
# This gives: (2000/16) × 16 × 100 epochs = 200,000 samples

conda activate clrernet
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

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
    --logs-dir ./logs/culane_2k_fair \
    -b 1 \
    -j 0 \
    --num-epochs 100 \
    --lr 0.0001 \
    --optim AdamW \
    -wd 0 \
    --accum-steps 16 \
    --save-steps 500
```

⚠️ **CRITICAL:** `--train-split train_2k` is required! Without it, defaults to full dataset (11,675 samples).

✅ **train.py supports `--accum-steps` (line 71)**
⚠️ **Note:** train.py only supports `AdamW`/`SGD` (not `adam`). Using `AdamW` with `-wd 0` approximates `adam`.

This matches LSTR's 200,000 total samples with proper batch normalization behavior.

---

## Summary: What to Match

| Priority | Parameter | Value to Use |
|----------|-----------|--------------|
| 🔴 Critical | Total samples | 200,000 (100 epochs) |
| 🔴 Critical | Learning rate | 0.0001 (1e-4) |
| 🔴 Critical | Optimizer | adam |
| 🟡 Important | Input size | Similar (~250k pixels) ✅ |
| 🟡 Important | Dataset | train_2k ✅ |
| 🟢 Nice to have | Augmentation | Match if possible |

**Current Status:**
- Your current training (50 epochs) will give UNFAIR comparison
- CondLSTR will have LOWER F1 because it sees half the data
- Need to retrain with 100 epochs + lr=1e-4 + adam optimizer

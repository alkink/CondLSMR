# Test Commands from README.md (Adapted for CULane)

> **IMPORTANT**: Test/Visualization requires a trained checkpoint. Either put `model_best.pth.tar` or `checkpoint.pth.tar` in `--logs-dir`, or pass a file with `--checkpoint /path/to/checkpoint.pth.tar`. Train first if you have no checkpoint.

> **OOM (CUDA out of memory)**: 12 GB GPU'da:
> - **Eğitim**: `--img-size 800 320`, `-b 1` ve `-p amp` kullanın (matcher/loss büyük tensor ile OOM alıyorsanız batch 1 + AMP gerekir).
> - **Test**: `-b 1` ve eğitimde img-size kullandıysanız testte de `--img-size 800 320` ekleyin.

> **DataLoader worker Segmentation fault**: Worker process segfault alıyorsanız `-j 0` kullanın (veri ana process'te yüklenir, daha yavaş ama stabil).

> **Segmentation fault (core dumped)**: Eval/train geçişi kaldırıldı (1x1 BN artık `F.batch_norm(..., training=False)` ile). Hâlâ segfault alırsanız: `PYTHONFAULTHANDLER=1 python tools/train.py ...` ile hata konumunu görün; AMP'siz deneyin: `-p fp32`.

## Pre-requisite: Install timeout_decorator
```bash
pip install timeout-decorator
```

---

## Commands from README.md (Section: Training & Evaluation)

### 1. Training Debug (CULane)
```bash
python tools/train.py -a CondLSTR2DRes34 -d culane -v v1.0 -c 21 -t lane_det_2d --data-dir /home/alki/projects/datasets --logs-dir ./logs/culane -b 4 -j 4
```

### 2. Evaluation Debug (CULane) - Requires checkpoint
```bash
python tools/train.py -a CondLSTR2DRes34 -d culane -v v1.0 -c 21 -t lane_det_2d --data-dir /home/alki/projects/datasets --logs-dir ./logs/culane -b 4 -j 4 --eval
```

### 3. Test/Visualization Debug (CULane) - Requires checkpoint
```bash
python tools/test.py -a CondLSTR2DRes34 -d culane -v v1.0 -c 21 -t lane_det_2d --data-dir /home/alki/projects/datasets --logs-dir ./logs/culane --test-dir ./output/culane -b 4 -j 4
```
**Or with a specific checkpoint file:**
```bash
python tools/test.py -a CondLSTR2DRes34 -d culane -v v1.0 -c 21 -t lane_det_2d --data-dir /home/alki/projects/datasets --logs-dir ./logs/culane --checkpoint /path/to/checkpoint.pth.tar --test-dir ./output/culane -b 4 -j 4
```

### 4. Multi-GPU Training (CULane)
```bash
python -m torch.distributed.launch --nproc_per_node=4 tools/train.py -a CondLSTR2DRes34 -d culane -v v1.0 -c 21 -t lane_det_2d --data-dir /home/alki/projects/datasets --logs-dir ./logs/culane --gpu-ids 0,1,2,3 -b 4 -j 4
```

### 5. Multi-GPU Evaluation (CULane) - Requires checkpoint
```bash
python -m torch.distributed.launch --nproc_per_node=4 tools/train.py -a CondLSTR2DRes34 -d culane -v v1.0 -c 21 -t lane_det_2d --data-dir /home/alki/projects/datasets --logs-dir ./logs/culane --gpu-ids 0,1,2,3 -b 4 -j 4 --eval
```

### 6. Multi-GPU Test/Visualization (CULane) - Requires checkpoint
```bash
python -m torch.distributed.launch --nproc_per_node=4 tools/test.py -a CondLSTR2DRes34 -d culane -v v1.0 -c 21 -t lane_det_2d --data-dir /home/alki/projects/datasets --logs-dir ./logs/culane --test-dir ./output/culane --gpu-ids 0,1,2,3 -b 4 -j 4
```

### 7. Resume from Checkpoint (CULane)
```bash
python -m torch.distributed.launch --nproc_per_node=4 tools/train.py -a CondLSTR2DRes34 -d culane -v v1.0 -c 21 -t lane_det_2d --data-dir /home/alki/projects/datasets --logs-dir ./logs/culane --gpu-ids 0,1,2,3 -b 4 -j 4 --resume
```

### 8. Training with AMP (CULane)
```bash
python -m torch.distributed.launch --nproc_per_node=4 tools/train.py -a CondLSTR2DRes34 -d culane -v v1.0 -c 21 -t lane_det_2d --data-dir /home/alki/projects/datasets --logs-dir ./logs/culane --gpu-ids 0,1,2,3 -b 4 -j 4 -p amp
```

### 9. Finetune with Pretrained Model (CULane)
```bash
python -m torch.distributed.launch --nproc_per_node=4 tools/train.py -a CondLSTR2DRes34 -d culane -v v1.0 -c 21 -t lane_det_2d --data-dir /home/alki/projects/datasets --logs-dir ./logs/culane --gpu-ids 0,1,2,3 -b 4 -j 4 --resume --load-model-only
```

---

## Command Parameter Changes (openlane → culane)

| Parameter | OpenLane (README) | CULane (Our Setup) |
|-----------|-------------------|-------------------|
| `-d` | `openlane` | `culane` |
| `-v` | `2d` | `v1.0` |
| `--data-dir` | `/path/to/datasets/` | `/home/alki/projects/datasets` |
| `--logs-dir` | `/path/to/checkpoint` | `./logs/culane` |
| `--test-dir` | `/path/to/output` | `./output/culane` |

---

## Train/Test with train_2k and test_2k (küçük subset)

CULane klasöründe `list/train_2k.txt` ve `list/test_2k.txt` (veya doğrudan `train_2k.txt` / `test_2k.txt`) varsa, ilk veri yüklemede bu listelerden `culane_infos_train_2k_*.pkl` ve `culane_infos_test_2k_*.pkl` otomatik üretilir. Eğitim ve testi bu split’lerle yapmak için:

**Eğitim (train_2k ile). 12 GB GPU için OOM önleme: `--img-size 800 320`, `-b 1`, `-p amp`, `-j 0`:**
```bash
python tools/train.py -a CondLSTR2DRes34 -d culane -v v1.0 -c 21 -t lane_det_2d --data-dir /home/alki/projects/datasets --logs-dir ./logs/culane_2k --train-split train_2k --val-split test_2k --img-size 800 320 -b 1 -j 0 -p amp --num-epochs 2
```

**Evaluation (test_2k üzerinde)** — eğitimde `--img-size` kullandıysanız burada da ekleyin:
```bash
python tools/train.py -a CondLSTR2DRes34 -d culane -v v1.0 -c 21 -t lane_det_2d --data-dir /home/alki/projects/datasets --logs-dir ./logs/culane_2k --train-split train_2k --val-split test_2k --img-size 800 320 -b 2 -j 2 --eval
```

**Test/Visualization (test_2k, OOM için -b 1 ve --img-size 800 320):**
```bash
python tools/test.py -a CondLSTR2DRes34 -d culane -v v1.0 -c 21 -t lane_det_2d --data-dir /home/alki/projects/datasets --logs-dir ./logs/culane_2k --test-dir ./output/culane_2k --split test_2k --img-size 800 320 -b 1 -j 2
```

---

## Quick Test Workflow

1. **First, install missing dependency:**
   ```bash
   pip install timeout-decorator
   ```

2. **Run a short training test (2 epochs)** — 12 GB GPU için `--img-size 800 320` ve train_2k:
   ```bash
   python tools/train.py -a CondLSTR2DRes34 -d culane -v v1.0 -c 21 -t lane_det_2d --data-dir /home/alki/projects/datasets --logs-dir ./logs/culane_test --train-split train_2k --val-split test_2k --img-size 800 320 -b 2 -j 2 --num-epochs 2
   ```

3. **Then run evaluation** (aynı --img-size):
   ```bash
   python tools/train.py -a CondLSTR2DRes34 -d culane -v v1.0 -c 21 -t lane_det_2d --data-dir /home/alki/projects/datasets --logs-dir ./logs/culane_test --train-split train_2k --val-split test_2k --img-size 800 320 -b 2 -j 2 --eval
   ```

4. **Then run test/visualization** (aynı --img-size, -b 1):
   ```bash
   python tools/test.py -a CondLSTR2DRes34 -d culane -v v1.0 -c 21 -t lane_det_2d --data-dir /home/alki/projects/datasets --logs-dir ./logs/culane_test --test-dir ./output/culane_test --split test_2k --img-size 800 320 -b 1 -j 2
   ```

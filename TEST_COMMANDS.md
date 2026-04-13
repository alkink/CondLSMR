# Test Commands - Run These in VSCode Terminal

## Step 1: Activate Conda Environment
```bash
conda activate hdmapnet
```

## Step 2: Navigate to Project Directory
```bash
cd /home/alki/projects/CondLSTR
```

## Step 3: Verify Dataset Paths
```bash
ls -la /home/alki/projects/datasets/culane/*.txt
```

Expected output should show symlinks for train_gt.txt, val.txt, test.txt

## Step 4: Install Missing Dependencies (if needed)
```bash
# Install jsonlines if not already installed
pip install jsonlines
```

## Step 5: Run Dataset Test Script
```bash
python test_dataset.py
```

This will:
- Check all required paths exist
- Verify dataset can be imported
- Test loading a sample image
- Verify model is available

## Step 6: Quick Dataset Check (Alternative)
If the test script fails, try these individual checks:

### Check dataset structure
```bash
head -3 /home/alki/projects/datasets/culane/train_gt.txt
```

### Check image file exists
```bash
ls -la /home/alki/projects/datasets/culane/driver_23_30frame/05151649_0422.MP4/ | head -10
```

### Check annotation file exists
```bash
cat /home/alki/projects/datasets/culane/driver_23_30frame/05151649_0422.MP4/00000.lines.txt
```

## Step 7: Test Data Loading (Python)
```bash
python -c "
from data.datasets.lane.culane import CULaneDataset
ds = CULaneDataset(root='/home/alki/projects/datasets/culane', split='train', version='v1.0')
print(f'Dataset size: {len(ds)}')
sample = ds[0]
print(f'Sample keys: {list(sample.keys())}')
"
```

**Note**: This will trigger preprocessing if pickle files don't exist, which may take a few minutes.

## Step 8: Verify Model Can Be Created
```bash
python -c "
from modeling import models
print('Available models:', models.names()[:10])
model = models.create('CondLSTR2DRes34', num_classes=21)
print('Model created successfully!')
"
```

## Step 9: Full Training Test (Dry Run)
```bash
python tools/train.py \
    -a CondLSTR2DRes34 \
    -d culane \
    -v v1.0 \
    -c 21 \
    -t lane_det_2d \
    --data-dir /home/alki/projects/datasets \
    --logs-dir ./logs/test_run \
    -b 1 \
    -j 2 \
    --eval-epoch 1 \
    --num-epochs 2
```

This will:
- Run for 2 epochs only
- Use small batch size
- Create log directory
- Validate everything works before full training

## Expected Results

### Successful Test Output
```
============================================================
CULane Dataset Setup Validation
============================================================
============================================================
Testing CULane Dataset Paths
============================================================

Dataset Root: /home/alki/projects/datasets/culane
Symlink points to: /home/alki/projects/CULane

--- Testing List Files ---
  train_gt.txt: ✓ (symlink)
    → 88881 entries
  val.txt: ✓ (symlink)
    → 9676 entries
  test.txt: ✓ (symlink)
    → 5120 entries

--- Testing Directories ---
  driver_23_30frame: ✓
  laneseg_label_w16: ✓

--- Testing Sample Data ---
  Sample image: ✓
  Sample annotation (.lines.txt): ✓
    → 4 lanes in annotation

============================================================
Summary
============================================================
  Paths:       ✓ PASS
  Dataset:     ✓ PASS
  Model:       ✓ PASS

  🎉 All tests passed! Ready for training.
============================================================
```

## Troubleshooting

### "No module named 'jsonlines'"
```bash
pip install jsonlines
```

### "ModuleNotFoundError: No module named 'cv2'"
```bash
pip install opencv-python
```

### "ModuleNotFoundError: No module named 'torch'"
```bash
pip install torch torchvision
```

### "No module named 'mmcv'"
```bash
pip install -U openmim
mim install mmcv
```

### If all else fails, install all requirements:
```bash
pip install -r requirements.txt
```

Once tests pass, start full training:

```bash
python tools/train.py \
    -a CondLSTR2DRes34 \
    -d culane \
    -v v1.0 \
    -c 21 \
    -t lane_det_2d \
    --data-dir /home/alki/projects/datasets \
    --logs-dir ./logs/culane_training \
    -b 4 \
    -j 4
```

## Full Training (After Tests Pass)

### Single GPU:

### Multi-GPU:
```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    tools/train.py \
    -a CondLSTR2DRes34 \
    -d culane \
    -v v1.0 \
    -c 21 \
    -t lane_det_2d \
    --data-dir /home/alki/projects/datasets \
    --logs-dir ./logs/culane_training \
    --gpu-ids 0,1,2,3 \
    -b 4 \
    -j 4
```

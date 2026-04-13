# CULane Dataset Setup Guide

## Dataset Structure

Your CULane dataset has been successfully prepared! Here's the current setup:

### Directory Structure
```
/home/alki/projects/datasets/culane (symlink to /home/alki/projects/CULane)
├── driver_23_30frame/          # Training images
├── driver_37_30frame/
├── driver_100_30frame/
├── driver_161_90frame/
├── driver_182_30frame/
├── driver_193_90frame/
├── laneseg_label_w16/          # Segmentation labels
├── list/                       # Original list files directory
├── train_gt.txt -> list/train_gt.txt    # Symlink for training list
├── val.txt -> list/val.txt              # Symlink for validation list
└── test.txt -> list/test.txt            # Symlink for test list
```

### What was done:
1. ✅ Created `/home/alki/projects/datasets` directory
2. ✅ Created symlink: `/home/alki/projects/datasets/culane` → `/home/alki/projects/CULane`
3. ✅ Created symlinks for required list files (train_gt.txt, val.txt, test.txt)
4. ✅ Verified dataset structure matches expectations

### Dataset Contents:
- **Training data**: ~88,881 samples (train_gt.txt)
- **Validation data**: ~9,676 samples (val.txt)
- **Test data**: Available in test.txt
- **Image format**: JPG files with corresponding .lines.txt annotation files
- **Annotation format**: Each .lines.txt file contains lane points (x, y coordinates)

## Next Steps

### 1. Activate the Conda Environment
```bash
conda activate hdmapnet
```

### 2. Preprocess the Dataset (Optional - will run automatically on first training)
The preprocessing will create pickle files for faster data loading:
```bash
cd /home/alki/projects/CondLSTR
python data/datasets/lane/culane/preprocess.py \
    --root /home/alki/projects/datasets/culane \
    --version v1.0 \
    --num-workers 8
```

This will create:
- `culane_infos_train_v1.0.pkl`
- `culane_infos_val_v1.0.pkl`

**Note**: The preprocessing will run automatically during the first training if these files don't exist, so this step is optional.

### 3. Start Training

#### Debug Training (Single GPU, Small Batch)
```bash
cd /home/alki/projects/CondLSTR
python tools/train.py \
    -a CondLSTR2DRes34 \
    -d culane \
    -v v1.0 \
    -c 21 \
    -t lane_det_2d \
    --data-dir /home/alki/projects/datasets \
    --logs-dir ./logs/culane_debug \
    -b 4 \
    -j 4
```

#### Multi-GPU Training (Recommended)
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
    --logs-dir ./logs/culane_training \
    --gpu-ids 0,1,2,3 \
    -b 4 \
    -j 4
```

### 4. Evaluation
```bash
cd /home/alki/projects/CondLSTR
python tools/train.py \
    -a CondLSTR2DRes34 \
    -d culane \
    -v v1.0 \
    -c 21 \
    -t lane_det_2d \
    --data-dir /home/alki/projects/datasets \
    --logs-dir ./logs/culane_training \
    -b 4 \
    -j 4 \
    --eval
```

### 5. Test/Visualization
```bash
cd /home/alki/projects/CondLSTR
python tools/test.py \
    -a CondLSTR2DRes34 \
    -d culane \
    -v v1.0 \
    -c 21 \
    -t lane_det_2d \
    --data-dir /home/alki/projects/datasets \
    --logs-dir ./logs/culane_training \
    --test-dir ./output/culane_results \
    -b 4 \
    -j 4
```

## Command Line Arguments Explained

- `-a` / `--arch`: Model architecture (CondLSTR2DRes34)
- `-d` / `--dataset`: Dataset name (culane)
- `-v` / `--version`: Dataset version (v1.0)
- `-c` / `--num-classes`: Number of classes (21 for lane detection)
- `-t` / `--task`: Task type (lane_det_2d for 2D lane detection)
- `--data-dir`: Parent directory containing the dataset (not the dataset itself!)
- `--logs-dir`: Directory to save checkpoints and logs
- `--test-dir`: Directory to save test results
- `-b` / `--batch-size`: Batch size per GPU
- `-j` / `--num-workers`: Number of data loading workers
- `--gpu-ids`: GPU IDs to use (for multi-GPU training)
- `--eval`: Run evaluation only
- `--resume`: Resume from checkpoint

## Important Notes

1. **Data Directory**: Use `--data-dir /home/alki/projects/datasets`, NOT the culane directory itself. The code will automatically append the dataset name to form the full path.

2. **First Training**: The first training run will take longer because it needs to preprocess the dataset and create pickle files. Subsequent runs will be much faster.

3. **Memory Requirements**: CULane is a large dataset. Make sure you have sufficient GPU memory. Adjust the batch size (`-b`) if you encounter out-of-memory errors.

4. **Number of Workers**: Adjust `-j` based on your CPU cores. More workers = faster data loading, but more memory usage.

5. **Checkpoints**: Checkpoints will be saved in the `--logs-dir` directory as:
   - `checkpoint.pth.tar` (latest checkpoint)
   - `model_best.pth.tar` (best model based on validation metric)

## Evaluation

CULane evaluation uses pure Python (no OpenCV C++ compilation needed). The evaluation tool is already included in the project.

### Run CULane Evaluation

```bash
cd /home/alki/projects/CondLSTR
python tools/metrics/lane/culane.py \
    --pred_dir ./output/culane_results \
    --anno_dir /home/alki/projects/datasets/culane \
    --list /home/alki/projects/datasets/culane/list/test.txt \
    --official \
    --width 30
```

### Run Evaluation on All Test Subsets

```bash
python tools/metrics/lane/culane.py \
    --pred_dir ./output/culane_results \
    --anno_dir /home/alki/projects/datasets/culane \
    --list /home/alki/projects/datasets/culane/list/test.txt \
          /home/alki/projects/datasets/culane/list/test_split/test0_normal.txt \
          /home/alki/projects/datasets/culane/list/test_split/test1_crowd.txt \
          /home/alki/projects/datasets/culane/list/test_split/test2_hlight.txt \
          /home/alki/projects/datasets/culane/list/test_split/test3_shadow.txt \
          /home/alki/projects/datasets/culane/list/test_split/test4_noline.txt \
          /home/alki/projects/datasets/culane/list/test_split/test5_arrow.txt \
          /home/alki/projects/datasets/culane/list/test_split/test6_curve.txt \
          /home/alki/projects/datasets/culane/list/test_split/test7_cross.txt \
          /home/alki/projects/datasets/culane/list/test_split/test8_night.txt \
    --official \
    --width 30
```

### Evaluation Arguments
- `--pred_dir`: Path to prediction results (output directory from test.py)
- `--anno_dir`: Path to CULane dataset
- `--list`: Path to test list file(s)
- `--official`: Use official CULane metric calculation method
- `--width`: Lane width for IoU calculation (default: 30 pixels)

**Note**: The "Install Evaluation Tools" section in README.md is for OpenLane dataset only. CULane uses pure Python evaluation, so no OpenCV C++ compilation is needed.

## Troubleshooting

### If you get "FileNotFoundError" for pickle files:
The preprocessing will run automatically. Just wait for it to complete.

### If you get CUDA out of memory:
Reduce batch size: `-b 2` or `-b 1`

### If training is too slow:
- Increase number of workers: `-j 8` or `-j 16`
- Use AMP (Automatic Mixed Precision): `-p amp`
- Use multiple GPUs as shown in Multi-GPU training section

### If you need to resume training:
Add `--resume` flag to your training command

## Dataset Statistics

- Total training samples: 88,881
- Total validation samples: 9,676
- Image resolution: Typically 1640x590 pixels
- Number of lanes per image: 1-4 lanes
- Annotation format: Ground truth lane points in .lines.txt files

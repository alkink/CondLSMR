# CondLSTR Test Komutları (Eğitim Sonrası)

## Checkpoint'ler Hazır ✅

```
logs/culane_2k_fair/
├── checkpoint.pth.tar      (338 MB - Son checkpoint)
├── model_best.pth.tar      (338 MB - En iyi model)
└── log.txt
```

---

## 1. Evaluation (Validation Set)

`model_best.pth.tar` ile validation/test set üzerinde evaluation:

```bash
conda activate clrernet

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
    --resume \
    --eval
```

Bu komut:
- `model_best.pth.tar`'ı yükler
- test_2k setinde evaluation yapar
- Sonuçları ekrana yazdırır

---

## 2. Test + Visualization (Predictions Kaydet)

Tahminleri kaydetmek için:

```bash
conda activate clrernet

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
    --test-dir ./output/culane_2k_fair \
    --resume \
    --test
```

Bu komut:
- `model_best.pth.tar`'ı yükler
- Tahminleri `./output/culane_2k_fair/` klasörüne kaydeder
- `.txt` formatında lane coordinates çıktılar

---

## 3. CULane Metrics (F1 Score)

CULane resmi metricleri ile F1 skoru hesaplamak için:

```bash
conda activate clrernet

# Önce test/visualization çalıştırın (yukarıdaki --test komutu)
# Sonra CULane metric scriptini çalıştırın:

python tools/metrics/lane/culane.py \
    --pred-dir ./output/culane_2k_fair \
    --anno-dir /home/alki/projects/datasets/culane/list
```

Bu çıktı:
- TP, FP, FN
- Precision, Recall, F1
- F1 score (LSTR ile karşılaştırma için)

---

## 4. Test Görüntüleri Görselleştirme

```bash
conda activate clrernet

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
    --test-dir ./output/culane_2k_fair_viz \
    --resume \
    --test \
    --show
```

`--show` flag'i ile lane tahminleri görüntü üzerine çizilir.

---

## Parametre Açıklamaları

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `--logs-dir` | ./logs/culane_2k_fair | Eğitim checkpoint konumu |
| `--resume` | - | model_best.pth.tar'ı yükler |
| `--eval` | - | Evaluation only, prediction kaydetmez |
| `--test` | - | Test mode, prediction kaydeder |
| `--show` | - | Lane çizimlerini görselleştirir |
| `--test-dir` | ./output/... | Prediction çıktı klasörü |
| `--train-split` | train_2k | Eğitim seti (checkpoint'ten okunur) |
| `--val-split` | test_2k | Test seti |

---

## LSTR vs CondLSTR Karşılaştırma

İki modelin F1 skorlarını karşılaştırmak için:

```bash
# CondLSTR F1 (test_2k set)
python tools/metrics/lane/culane.py \
    --pred-dir ./output/culane_2k_fair \
    --anno-dir /home/alki/projects/datasets/culane/list

# LSTR F1 (LSTR projesindeki predictions ile)
cd /home/alki/projects/LSTR
python tools/metrics/lane/culane.py \
    --pred-dir ./results/culane_2k \
    --anno-dir /home/alki/projects/datasets/culane/list
```

Her iki modelin de aynı test seti (test_2k) üzerinde değerlendirildiğinden emin olun.

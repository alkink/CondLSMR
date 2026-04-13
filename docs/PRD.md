# PRD – CondLSTR Proje Değişiklik Kayıtları

## 2025-02-19: Checkpoint Dokümantasyonu ve Test İyileştirmesi

### Bağlam
- Kullanıcı: CondLSTR için önceden eğitilmiş (SOTA) checkpoint'lerin nerede olduğunu ve modelin nasıl test edileceğini sordu.
- README ve orijinal repo incelendi; resmi checkpoint indirme linki bulunmuyor.

### Yapılan Değişiklikler

1. **README.md**
   - "Pre-trained Checkpoints" bölümü eklendi.
   - Durum özeti: Resmi checkpoint yayını yok; kendi eğitim, checkpoint konumu ve test için `--checkpoint` kullanımı açıklandı.
   - Orijinal repo ve Issues linki eklendi.

2. **tools/test.py**
   - Yeni argüman: `--checkpoint PATH` — Doğrudan bir `.pth.tar` checkpoint dosyası yolu verilebiliyor (önceki davranış: sadece `--logs-dir` altında sabit dosya adları).
   - Checkpoint arama sırası: `--checkpoint` (dosya) → `--logs-dir/model_best.pth.tar` → `--logs-dir/checkpoint.pth.tar`.
   - Hata mesajı "Checkpoint ... does not exist!" olacak şekilde güncellendi.

3. **docs/PRD.md**
   - Bu değişiklik kaydı oluşturuldu.

### Test Nasıl Yapılır (Özet)

- **Checkpoint `--logs-dir` içindeyse (örn. `./logs/culane`):**
  ```bash
  python tools/test.py -a CondLSTR2DRes34 -d culane -v v1.0 -c 21 -t lane_det_2d --data-dir /path/to/datasets --logs-dir ./logs/culane --test-dir ./output/culane -b 4 -j 4
  ```
- **Checkpoint tek bir dosyadaysa:**
  ```bash
  python tools/test.py -a CondLSTR2DRes34 -d culane -v v1.0 -c 21 -t lane_det_2d --data-dir /path/to/datasets --logs-dir ./logs/culane --checkpoint /path/to/checkpoint.pth.tar --test-dir ./output/culane -b 4 -j 4
  ```
- Detaylı komutlar için `TEST_COMMANDS_README.md` kullanılabilir.

### Notlar
- Önceden eğitilmiş model olmadan denemek için önce kısa bir eğitim (örn. 2 epoch) yapıp çıkan checkpoint ile test edilebilir (TEST_COMMANDS_README.md'deki Quick Test Workflow).

---

## 2025-02-19: train_2k / test_2k ve OOM Düzenlemeleri

### Bağlam
- Kullanıcı: Checkpoint olmadığı için train yapacak; eğitimi tam train seti yerine **train_2k**, testi **test_2k** ile yapmak istiyor.
- Test sırasında CUDA OOM (15.26 GiB isteği, 12 GB GPU) alındı.

### Yapılan Değişiklikler

1. **data/datasets/lane/culane/preprocess.py**
   - `DATASETS_TRAIN_2K` ve `DATASETS_VAL_2K` eklendi: `list/train_2k.txt`, `list/test_2k.txt` (yoksa `train_2k.txt`, `test_2k.txt` deneniyor).
   - `lane_data_prep` içinde bu list dosyaları varsa `culane_infos_train_2k_<version>.pkl` ve `culane_infos_test_2k_<version>.pkl` üretiliyor.
   - `_resolve_list_path` ve `_build_infos_from_list` helper'ları eklendi.

2. **data/datasets/lane/culane/culane.py**
   - `test_mode`: `split in ('test', 'test_2k')` olacak şekilde güncellendi.

3. **tools/train.py**
   - `--train-split` (varsayılan: `train`) ve `--val-split` (varsayılan: `val`) eklendi.
   - Eğitim/val dataset'leri bu split'lerle oluşturuluyor (örn. `--train-split train_2k --val-split test_2k`).

4. **tools/test.py**
   - Zaten `--split` var (varsayılan `val`); test_2k için `--split test_2k` kullanılıyor.

5. **TEST_COMMANDS_README.md**
   - OOM uyarısı: 12 GB GPU'da test için `-b 1` veya `-b 2` önerisi eklendi.
   - "Train/Test with train_2k and test_2k" bölümü ve örnek komutlar eklendi.
   - Quick Test Workflow, isteğe bağlı `--train-split train_2k --val-split test_2k` ve test için `--split test_2k`, `-b 1` ile güncellendi.

### Kullanım Özeti (train_2k / test_2k)

- CULane root'ta `list/train_2k.txt` ve `list/test_2k.txt` (veya root'ta `train_2k.txt`, `test_2k.txt`) olmalı. İlk veri yüklemede bu listelerden pkl'ler otomatik üretilir.
- Eğitim: `--train-split train_2k --val-split test_2k`
- Test: `--split test_2k` ve OOM riski için `-b 1` veya `-b 2`.

---

## 2025-02-19: Collate hatası (tensor size 4 vs 3 – kanal uyumsuzluğu)

### Bağlam
- Eğitim sırasında `RuntimeError: The size of tensor a (4) must match the size of tensor b (3) at non-singleton dimension 0` alındı.
- Bazı görüntüler 4 kanal (RGBA), bazıları 3 kanal (RGB); batch’te kanal sayısı eşleşmiyordu.

### Yapılan Değişiklikler

1. **data/datasets/utils/loading.py (LoadImageFromFile)**
   - `mmcv.imread` sonrası: `img.ndim == 3` ve `img.shape[2] == 4` ise `img = img[:, :, :3].copy()` ile 3 kanala indiriliyor.

2. **data/transforms/lane/utils.py (nested_tensor_from_tensor_list)**
   - Collate’da yedek: tensor listesinde kanal sayıları farklıysa ve 3 varsa, hepsi 3 kanala kesiliyor (`t[:3]`).

---

## 2025-02-19: Eğitim OOM – img-size

- 12 GB GPU'da eğitim OOM için `--img-size 800 320` ve `-b 2` eklendi (train.py, test.py, culane_transforms kwargs). TEST_COMMANDS_README güncellendi.

---

## 2025-02-19: Evaluation hatası – data_dict['lane_points'] / lane_attris

### Bağlam
- Epoch 0 bittikten sonra evaluation aşamasında `lane_det_2d.py` içinde `data_dict['lane_points']` KeyError oluşuyordu.
- Sorun: `ToTensor` transform'u sadece `keys` ve `meta_keys` içindeki key'leri koruyordu; `lane_points` ve `lane_attris` bu listelerde olmadığı için kayboluyordu.
- CULane veri setinde `lane_attris` yok; metrik hem `lane_points` hem `lane_attris` bekliyor. Ayrıca metrik tgt için `.cpu().numpy()` kullanıyordu, batch’te numpy/list de gelebiliyor.

### Yapılan Değişiklikler

1. **data/transforms/lane/transforms.py (GenerateLaneLine2D)**
   - `lane_attris` yoksa (CULane gibi) varsayılan ekleniyor: `results['lane_attris'] = np.zeros(len(lane_points), dtype=np.int64)`.

2. **modeling/metrics/lane/lane_det_2d.py (LaneDet2DMetric.add_batch)**
   - Hedef (tgt) nokta ve attribute’ler için tensor/numpy uyumluluğu: `to_numpy_points(x)` ve `to_int_attr(a)` ile hem tensor hem numpy/list kabul ediliyor; böylece collate’dan gelen list/numpy batch’ler evaluation’da hata vermiyor.

3. **data/transforms/transforms.py (ToTensor)**
   - `ToTensor` daha önce sadece `keys` ve `meta_keys` içindeki alanları döndürdüğü için `lane_points` / `lane_attris` kaybolabiliyordu.
   - `keys` ve `meta_keys` dışında kalan alanları da `data` dict’ine kopyalayacak şekilde güncellendi; böylece evaluation/training için gerekli ek alanlar korunuyor.

---

## 2025-02-19: Eğitim – lane_attris numpy / mask-attr uzunluk uyumsuzluğu

### Bağlam
- Eğitim sırasında `loss.py` içinde `gt_attr.long()` çağrısında `AttributeError: 'numpy.ndarray' object has no attribute 'long'` alındı.
- Sonrasında `assert len(gt_mask) == len(gt_attr)` hatası görüldü. Sebep: `img_mask` collate sırasında **M_max** lane kanalına padleniyor; `lane_attris` ise örnek başına **M_i** (gerçek lane sayısı) ile geliyor.

### Yapılan Değişiklikler
1. **modeling/models/detectors/lane/cond_lstr_2d/loss.py**
   - `gt_attr` tensor değilse `torch.from_numpy(np.asarray(gt_attr, dtype=np.int64)).to(device)` ile tensöre çevrilip `.long()` uygulanıyor.
   - `gt_mask` ve `gt_attr` lane sayısı uyuşmazlığında, ortak güvenli uzunluk \(n\_lanes = min(len(gt_mask), len(gt_attr))\) ile ikisi de aynı uzunluğa kesiliyor.

---

## 2025-02-19: Stabilite – OpenCV thread/OpenCL kapatma

### Bağlam
- Eğitim sırasında nadiren `Segmentation fault (core dumped)` görülebiliyor (çoğunlukla OpenCV/imgaug/CUDA etkileşimi).

### Yapılan Değişiklik
1. **data/transforms/lane/transforms.py**
   - `cv2.setNumThreads(0)` ve `cv2.ocl.setUseOpenCL(False)` eklendi (try/except ile), stabiliteyi artırmak için.

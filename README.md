# YOLOv8 Face Mask Detection

Object detection uçtan uca projesi: yüzlerde maske takılı/takılı değil/yanlış takılı
sınıflarını YOLOv8n ile tespit eden bir model. Pascal VOC formatındaki bir veri
setinin YOLO formatına dönüştürülmesinden eğitim sonuçlarının yorumlanmasına kadar
tüm pipeline'ı kapsar.

## Veri Seti

[Kaggle: andrewmvd/face-mask-detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
— 853 görsel, Pascal VOC XML formatında etiketlenmiş, 3 sınıf:

| Sınıf                   | Instance sayısı | Oran   |
|-------------------------|----------------:|-------:|
| `with_mask`             |            3232 |  77.6% |
| `without_mask`          |             717 |  17.2% |
| `mask_weared_incorrect` |             123 |   2.9% |

Sınıf dağılımı ciddi şekilde dengesiz — bu durum aşağıdaki sonuçlarda doğrudan
görülüyor.

## Pipeline

1. **`scripts/convert_and_split.py`** — Pascal VOC XML etiketlerini (mutlak
   köşe koordinatı: xmin/ymin/xmax/ymax) YOLO formatına (normalize edilmiş
   merkez + genişlik/yükseklik) dönüştürür, aynı adımda train/val split'i
   yapar (%80/%20, `seed=42`, 683/170 görsel).
2. **`dataset/data.yaml`** — YOLO eğitim konfigürasyonu (sınıf isimleri, split
   yolları).
3. **Eğitim** — `yolov8n`, 30 epoch, `imgsz=416`, CPU üzerinde (~36 dk).
4. **Değerlendirme** — precision/recall/mAP, confusion matrix, PR/F1
   eğrileri (`runs/detect/runs/mask_detect/`).
5. **Inference testi** — eğitilmiş modelle örnek görseller üzerinde tahmin
   (`runs/detect/runs/inference_test/`).

## Sonuçlar

Genel (tüm sınıflar, `runs/detect/runs/mask_detect/results.csv`, epoch 30):

| Precision | Recall | mAP50 | mAP50-95 |
|----------:|-------:|------:|---------:|
|      0.87 |   0.67 |  0.75 |     0.52 |

Sınıf bazında:

| Sınıf                   | Precision | Recall | mAP50 | mAP50-95 |
|-------------------------|----------:|-------:|------:|---------:|
| `with_mask`             |     0.958 |  0.914 | 0.957 |    0.668 |
| `without_mask`          |     0.921 |  0.613 | 0.741 |    0.482 |
| `mask_weared_incorrect` |     0.754 |  0.474 | 0.541 |    0.409 |

**Bu sonuçlar cherry-pick edilmemiştir** — model en çok veriye sahip
`with_mask` sınıfında güçlü (mAP50 0.96), en az veriye sahip
`mask_weared_incorrect` sınıfında zayıf (mAP50 0.54). Genel recall'ın (0.67)
precision'dan (0.87) düşük olmasının temel sebebi, azınlık sınıflarda
modelin nesneleri kaçırması (düşük recall) — özellikle kalabalık/kısmi
görünür yüzlerde. Bu, veri setindeki class imbalance'ın (77.6% / 17.2% / 2.9%)
doğrudan bir sonucu.

## Sonraki adımlar

- Class imbalance için veri artırma veya class-weighted loss
- Daha büyük model (`yolov8s`/`yolov8m`) veya daha fazla epoch ile karşılaştırma
- Kendi görsel/webcam ile canlı inference testi

## Kurulum / Yeniden Üretme

Model ağırlıkları (`*.pt`) ve ham/işlenmiş görsel veri (`dataset/raw/`,
`dataset/images/`) repo boyutunu şişirmemek için `.gitignore` ile hariç
tutulmuştur. Yeniden üretmek için:

```bash
# 1. Veri setini indirip dataset/raw/extracted/{images,annotations} altına çıkar
# 2. XML -> YOLO dönüşümü + split
python scripts/convert_and_split.py

# 3. Eğitim
yolo detect train model=yolov8n.pt data=dataset/data.yaml epochs=30 imgsz=416
```

## Proje Yapısı

```
scripts/convert_and_split.py   # XML -> YOLO dönüşüm + train/val split
dataset/data.yaml              # YOLO eğitim konfigürasyonu
dataset/labels/                # YOLO formatında etiketler (train/val)
runs/detect/runs/mask_detect/  # Eğitim çıktıları: metrikler, curve'ler, confusion matrix
runs/detect/runs/inference_test/  # Örnek inference çıktıları
```

# Kidney Stone Segmentation with Meta SAM (ViT-B Fine-Tuning)

*[English](#english) | [Türkçe](#türkçe)*

---

<a name="english"></a>
## English

### Overview
This repository contains our deep learning scripts for **Kidney Stone Segmentation** built by fine-tuning Meta AI's **Segment Anything Model (`SAM ViT-B`)** on medical imaging datasets. It implements a two-stage training strategy: initial dataset verification (`kontrol.py`), base fine-tuning (`train.py`), and low-learning-rate secondary fine-tuning (`finetune_second.py`, `LR = 5e-6`, 10 epochs) on top of the best checkpoint (`best_kidney_model.pth`) to adapt the model across different kidney stone datasets (including Kaggle and MultiStone).

### Development Timeline & Process
- **November 26, 2025:** Built `kontrol.py` to inspect dataset composition (JPG/PNG/TXT counts) and detect whether source annotations were object-detection bounding boxes or pixel-level segmentation masks.
- **November 27, 2025:** Developed `finetune_second.py` for domain-adaptive second-stage fine-tuning (`best_kidney_model.pth` → Kaggle dataset) with 1024×1024 SAM preprocessing.
- **December 5, 2025:** Updated `train.py` (which was later also adapted as the template for our ISIC 2018 Melanoma SAM fine-tuning pipeline).

### Repository Contents
- `kontrol.py` — Dataset structure and annotation-format validator (JPG/PNG/TXT check).
- `train.py` — Base SAM (`vit_b`) fine-tuning script with 1024×1024 normalization and binary mask preprocessing.
- `finetune_second.py` — Second-stage low-LR (`5e-6`) fine-tuning script loading `best_kidney_model.pth`.
- `requirements.txt` — Python dependencies.

### Setup & Excluded Large Files
Large medical datasets (`Kidney_Stone_Dataset`, `MultiStone`, `YOLO_Ready_Dataset`, `data/`) and ~375 MB `.pth` checkpoints (`sam_vit_b_01ec64.pth`, `best_kidney_model.pth`, `last_kidney_model.pth`) are excluded for GitHub compatibility.
1. Download the official `sam_vit_b_01ec64.pth` checkpoint from Meta's Segment Anything repository.
2. Update `CHECKPOINT_PATH` and `DATA_ROOT` in `train.py` / `finetune_second.py` to match your local dataset paths.
3. Install dependencies and run:
```bash
pip install -r requirements.txt
python kontrol.py
python finetune_second.py
```

### Acknowledgments & Open-Source Credits
We thank the open-source researchers and dataset contributors who made this project possible:
- **[Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)** — Kirillov et al., Meta AI Research (2023).
- **Kaggle Kidney Stone & MultiStone Datasets** — Community-contributed medical imaging datasets.
- **[PyTorch](https://pytorch.org/)** & **[OpenCV](https://opencv.org/)**.

---

<a name="türkçe"></a>
## Türkçe

### Genel Bakış
Bu repo, Meta AI'ın **Segment Anything Model (`SAM ViT-B`)** mimarisini böbrek taşı medikal görüntüleri üzerinde ince ayar (fine-tuning) yaparak **Böbrek Taşı Segmentasyonu** gerçekleştiren eğitim ve veri kontrol betiklerini içerir. Proje; veri seti format kontrolü (`kontrol.py`), temel eğitim (`train.py`) ve en iyi ağırlık (`best_kidney_model.pth`) üzerinden düşük öğrenme oranıyla (`LR = 5e-6`) ikinci aşama ince ayar (`finetune_second.py`) adımlarını kapsar.

### Geliştirme Süreci ve Tarihçesi
- **26 Kasım 2025:** Veri setindeki görüntü (JPG), maske (PNG) ve etiket (TXT) sayılarını doğrulayan `kontrol.py` yazıldı.
- **27 Kasım 2025:** Eğitilmiş ilk böbrek taşı modeli (`best_kidney_model.pth`) üzerine ikinci aşama domain uyarlama yapan `finetune_second.py` geliştirildi.
- **5 Aralık 2025:** `train.py` güncellendi (bu yapı daha sonra Melanom SAM projemize de temel oluşturdu).

### Kurulum ve Hariç Tutulan Dosyalar
Büyük veri setleri (`Kidney_Stone_Dataset`, `MultiStone`, `YOLO_Ready_Dataset`) ve 375 MB boyutundaki `.pth` model ağırlıkları GitHub sınırları gereği hariç tutulmuştur. Meta SAM `sam_vit_b_01ec64.pth` dosyasını indirip betiklerdeki dosya yollarını güncelleyerek çalıştırabilirsiniz.

### Açık Kaynak Teşekkürler ve Kaynaklar
- **[Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)** — Kirillov vd., Meta AI Research (2023).
- **Kaggle Böbrek Taşı ve MultiStone Veri Setleri**.
- **[PyTorch](https://pytorch.org/)** ve **[OpenCV](https://opencv.org/)**.
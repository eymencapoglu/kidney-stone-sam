import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry

# -----------------------------------------------------------------------------
# 1. AYARLAR
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_TYPE = "vit_b"

# KRİTİK NOKTA: Sıfırdan başlamıyoruz, senin eğittiğin modeli yüklüyoruz
CHECKPOINT_PATH = r"C:\Users\Eymen\Desktop\Bobrek_Tasi_Projesi\best_kidney_model.pth"

# Hedef Veri Seti (Kaggle Verisi)
DATA_ROOT = r"C:\Users\Eymen\Desktop\Bobrek_Tasi_Projesi\data"

BATCH_SIZE = 1      
EPOCHS = 10         # 10 Epoch yeterli olacaktır
LR = 5e-6           # ÇOK DÜŞÜK HIZ (İnce ayar için)

# -----------------------------------------------------------------------------
# 2. YARDIMCI FONKSİYONLAR
# -----------------------------------------------------------------------------
def preprocess_image(image, target_size=1024):
    image = cv2.resize(image, (target_size, target_size))
    if len(image.shape) == 2:
        image = np.stack((image,)*3, axis=-1)
    image = (image - image.min()) / np.clip(image.max() - image.min(), a_min=1e-8, a_max=None) 
    image = (image * 255).astype(np.uint8)
    return image

def preprocess_mask(mask, target_size=1024):
    mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    mask = mask > 0  
    return mask.astype(np.uint8)

# -----------------------------------------------------------------------------
# 3. DATASET CLASS (Kaggle Formatına Özel)
# -----------------------------------------------------------------------------
class KaggleFineTuneDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.images = []
        self.masks = []
        
        if not os.path.exists(root_dir):
            print(f"HATA: '{root_dir}' klasörü bulunamadı!")
            return

        all_files = os.listdir(root_dir)
        # Sadece jpg/jpeg resimleri al
        image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg')) and '_mask' not in f]
        
        print(f"Klasör taranıyor: {root_dir}")
        
        for img_name in image_files:
            base_name = os.path.splitext(img_name)[0]
            
            # Kaggle setinde maskeler genelde "resimadi.png" şeklindedir
            # Ama biz yine de her iki ihtimali (resimadi_mask.png) de kontrol edelim.
            mask_path = None
            path1 = os.path.join(root_dir, base_name + ".png")
            path2 = os.path.join(root_dir, base_name + "_mask.png")
            
            if os.path.exists(path1): mask_path = path1
            elif os.path.exists(path2): mask_path = path2
            
            if mask_path:
                self.images.append(os.path.join(root_dir, img_name))
                self.masks.append(mask_path)

        print(f"Eşleşen Veri Sayısı: {len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.masks[idx], 0)

        if image is None or mask is None:
            return torch.zeros((3, 1024, 1024)), torch.zeros((1, 1024, 1024)), torch.tensor([0,0,1024,1024]).float()

        image_1024 = preprocess_image(image)
        mask_1024 = preprocess_mask(mask)

        y_indices, x_indices = np.where(mask_1024 > 0)
        
        if len(y_indices) > 0:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            H, W = mask_1024.shape
            
            # Fine-tune sırasında kutuyu çok az oynatıyoruz (Daha hassas öğrensin)
            x_min = max(0, x_min - np.random.randint(0, 5))
            x_max = min(W, x_max + np.random.randint(0, 5))
            y_min = max(0, y_min - np.random.randint(0, 5))
            y_max = min(H, y_max + np.random.randint(0, 5))
            
            bbox = np.array([x_min, y_min, x_max, y_max])
        else:
            bbox = np.array([0, 0, 1024, 1024]) 

        image_tensor = torch.tensor(image_1024).permute(2, 0, 1).float() 
        mask_tensor = torch.tensor(mask_1024).float().unsqueeze(0) 
        bbox_tensor = torch.tensor(bbox).float()

        return image_tensor, mask_tensor, bbox_tensor

# -----------------------------------------------------------------------------
# 4. EĞİTİM DÖNGÜSÜ
# -----------------------------------------------------------------------------
def main():
    print("--------------------------------------------------")
    print(f"İşlem Birimi: {DEVICE}")
    print("MOD: Kaggle Fine-Tuning (İnce Ayar)")
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"HATA: '{CHECKPOINT_PATH}' bulunamadı! Lütfen önceki eğitimin çıktısını buraya koyun.")
        return

    print(f"Önceki model yükleniyor: {CHECKPOINT_PATH}")
    
    # Modeli Başlat
    medsam_model = sam_model_registry[MODEL_TYPE](checkpoint=None) # Boş model oluştur
    # Ağırlıkları Yükle
    medsam_model.load_state_dict(torch.load(CHECKPOINT_PATH))
    medsam_model.to(DEVICE)
    medsam_model.train()

    # Encoder dondur
    for name, param in medsam_model.named_parameters():
        if name.startswith("image_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad = False
        else:
            param.requires_grad = True 

    # Optimizer (Düşük Learning Rate)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, medsam_model.parameters()), lr=LR, weight_decay=0.01)
    seg_loss = torch.nn.BCEWithLogitsLoss()

    # Dataset
    dataset = KaggleFineTuneDataset(DATA_ROOT)
    if len(dataset) == 0:
        return
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print("İnce Ayar Eğitimi Başlıyor...")

    for epoch in range(EPOCHS):
        epoch_loss = 0
        for i, (images, gt_masks, boxes) in enumerate(dataloader):
            images = images.to(DEVICE)
            gt_masks = gt_masks.to(DEVICE)
            boxes = boxes.to(DEVICE)
            
            optimizer.zero_grad()

            with torch.no_grad():
                image_embedding = medsam_model.image_encoder(images)
                sparse, dense = medsam_model.prompt_encoder(
                    points=None,
                    boxes=boxes[:, None, :], 
                    masks=None,
                )

            low_res_masks, _ = medsam_model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=medsam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )

            gt_resized = F.interpolate(gt_masks, size=(256, 256), mode='nearest')
            loss = seg_loss(low_res_masks, gt_resized)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if i % 50 == 0:
                 print(f"Epoch {epoch+1}, Adım {i}, Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"--- Epoch {epoch+1}/{EPOCHS} Bitti. Ort. Loss: {avg_loss:.4f} ---")
        
        # Her epoch sonu modeli güncelle
        torch.save(medsam_model.state_dict(), "final_kaggle_model.pth")
        print("Model güncellendi: final_kaggle_model.pth")
    
    print("Tebrikler! Final modeli hazır.")

if __name__ == "__main__":
    main()
import os
import glob
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

# BURAYA DİKKAT: İndirdiğin .pth dosyasının tam yolunu yaz
# Örnek: "C:\Users\Eymen\Desktop\Melanoma\sam_vit_b_01ec64.pth"
CHECKPOINT_PATH = r"C:\Users\Eymen\Desktop\Melanoma\sam_vit_b_01ec64.pth" 

# Train klasörlerinin yolları
TRAIN_IMG_DIR = r"C:\Users\Eymen\Desktop\Melanoma\data\images\train"
TRAIN_MASK_DIR = r"C:\Users\Eymen\Desktop\Melanoma\data\annotations\train"

BATCH_SIZE = 1      
EPOCHS = 25         
LR = 1e-5

print(f"Cihaz: {DEVICE}")

# -----------------------------------------------------------------------------
# 2. YARDIMCI FONKSİYONLAR
# -----------------------------------------------------------------------------
def preprocess_image(image, target_size=1024):
    # SAM 1024x1024 bekler
    image = cv2.resize(image, (target_size, target_size))
    if len(image.shape) == 2:
        image = np.stack((image,)*3, axis=-1)
    # Normalizasyon
    image = (image - image.min()) / np.clip(image.max() - image.min(), a_min=1e-8, a_max=None) 
    image = (image * 255).astype(np.uint8)
    return image

def preprocess_mask(mask, target_size=1024):
    mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    mask = mask > 0  # Binary yap
    return mask.astype(np.uint8)

# -----------------------------------------------------------------------------
# 3. MELANOMA DATASET CLASS (GÜNCELLENDİ)
# -----------------------------------------------------------------------------
class MelanomaDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        
        # Resim dosyalarını bul (jpg, png, jpeg)
        self.images = glob.glob(os.path.join(img_dir, "*.*"))
        
        print(f"Klasörde bulunan resim sayısı: {len(self.images)}")
        
        # Basit bir kontrol: Maske klasörü var mı?
        if not os.path.exists(mask_dir):
            print(f"HATA: Maske klasörü bulunamadı -> {mask_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        filename = os.path.basename(img_path)
        
        # Maske yolunu oluştur (Dosya isminin aynı olduğunu varsayıyoruz)
        # Eğer maskeler .png ama resimler .jpg ise uzantıyı değiştirmeliyiz:
        # mask_filename = os.path.splitext(filename)[0] + ".png"
        mask_filename = filename 
        mask_path = os.path.join(self.mask_dir, mask_filename)
        
        # Resim yükle
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # SAM RGB ister

        # Maske yükle
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            # Maske yoksa (bazen eşleşme hatası olabilir) siyah maske dön
            # print(f"Uyarı: Maske bulunamadı {filename}")
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Preprocess (1024x1024'e çevir)
        image_1024 = preprocess_image(image)
        mask_1024 = preprocess_mask(mask)

        # Bounding Box bul (Prompt için)
        y_indices, x_indices = np.where(mask_1024 > 0)
        
        if len(y_indices) > 0:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            
            # Kutuya biraz rastgele pay ekle (Augmentation gibi)
            H, W = mask_1024.shape
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))
            
            bbox = np.array([x_min, y_min, x_max, y_max])
        else:
            # Maske boşsa tam ekran kutu ver (veya 0,0,1,1)
            bbox = np.array([0, 0, 1024, 1024]) 

        # Tensor'a çevir
        # SAM (3, 1024, 1024) ister, Permute ile (H, W, C) -> (C, H, W) yapıyoruz
        image_tensor = torch.tensor(image_1024).permute(2, 0, 1).float() 
        mask_tensor = torch.tensor(mask_1024).float().unsqueeze(0) 
        bbox_tensor = torch.tensor(bbox).float()

        return image_tensor, mask_tensor, bbox_tensor

# -----------------------------------------------------------------------------
# 4. EĞİTİM DÖNGÜSÜ
# -----------------------------------------------------------------------------
def main():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"KRİTİK HATA: '{CHECKPOINT_PATH}' bulunamadı! Lütfen yolu kontrol edin.")
        return

    print("Model yükleniyor (Native SAM)...")
    sam_model = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam_model.to(DEVICE)
    sam_model.train()

    # Image Encoder ve Prompt Encoder'ı dondur (Sadece Mask Decoder eğitilsin)
    for name, param in sam_model.named_parameters():
        if name.startswith("image_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad = False
        else:
            param.requires_grad = True 

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, sam_model.parameters()), lr=LR, weight_decay=0.01)
    seg_loss = torch.nn.BCEWithLogitsLoss()

    # Dataset ve DataLoader
    dataset = MelanomaDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
    
    if len(dataset) == 0:
        print("Dataset boş görünüyor, yolları kontrol et.")
        return
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Eğitim Başlıyor... Toplam {len(dataset)} görüntü.")
    
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for i, (images, gt_masks, boxes) in enumerate(dataloader):
            images = images.to(DEVICE)
            gt_masks = gt_masks.to(DEVICE)
            boxes = boxes.to(DEVICE)
            
            optimizer.zero_grad()

            with torch.no_grad():
                # Görüntüyü encode et
                image_embedding = sam_model.image_encoder(images)
                
                # Prompt (Kutu) encode et
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None,
                    boxes=boxes[:, None, :], 
                    masks=None,
                )

            # Maskeyi tahmin et
            low_res_masks, _ = sam_model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            # Ground Truth maskeyi modelin çıktısına (256x256) uydur
            gt_masks_resized = F.interpolate(gt_masks, size=(256, 256), mode='nearest')
            
            # Loss hesapla
            loss = seg_loss(low_res_masks, gt_masks_resized)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if i % 10 == 0:
                 print(f"Epoch {epoch+1}, Adım {i}, Loss: {loss.item():.4f}")

        print(f"--- Epoch {epoch+1}/{EPOCHS} Tamamlandı. Ort. Loss: {epoch_loss/len(dataloader):.4f} ---")
        
        # Her epoch sonu kaydet
        torch.save(sam_model.state_dict(), "melanoma_sam_finetuned.pth")
    
    print("Eğitim Tamamlandı! Model 'melanoma_sam_finetuned.pth' olarak kaydedildi.")

if __name__ == "__main__":
    main()
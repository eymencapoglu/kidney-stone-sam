import os

# Senin klasör yolun (Loglardan aldım)
klasor_yolu = r"C:\Users\Eymen\Desktop\Bobrek_Tasi_Projesi\data"

print(f"Kontrol edilen klasör: {klasor_yolu}")

if os.path.exists(klasor_yolu):
    dosyalar = os.listdir(klasor_yolu)
    print(f"Toplam dosya sayısı: {len(dosyalar)}")
    
    # İlk 10 dosyayı yazdır
    print("\n--- İlk 10 Dosya ---")
    for dosya in dosyalar[:10]:
        print(dosya)
        
    # Uzantı sayımı yap
    jpg_sayisi = sum(1 for f in dosyalar if f.endswith('.jpg') or f.endswith('.jpeg'))
    png_sayisi = sum(1 for f in dosyalar if f.endswith('.png'))
    txt_sayisi = sum(1 for f in dosyalar if f.endswith('.txt'))
    
    print("\n--- İstatistik ---")
    print(f"JPG/JPEG (Resim): {jpg_sayisi}")
    print(f"PNG (Maske): {png_sayisi}")
    print(f"TXT (Etiket): {txt_sayisi}")
    
    if png_sayisi == 0 and txt_sayisi > 0:
        print("\nUYARI: Klasörde hiç PNG (maske) yok, sadece TXT var!")
        print("Bu bir 'Object Detection' veriseti olabilir. Maskeye çevirmemiz gerekecek.")
else:
    print("Klasör bulunamadı!")
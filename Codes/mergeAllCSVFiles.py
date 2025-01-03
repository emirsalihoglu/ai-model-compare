import pandas as pd
import os

# CSV dosyalarının bulunduğu klasör
csv_folder = r'C:\Web_Scraping\PubMed_Cancer_Articles\All Cancer CSV Files'

# Birleştirilmiş veri için boş bir DataFrame
combined_df = pd.DataFrame()

# Klasördeki tüm CSV dosyalarını döngüye al
for file in os.listdir(csv_folder):
    if file.endswith('.csv'):
        # Dosya yolunu belirle
        file_path = os.path.join(csv_folder, file)
        
        # CSV dosyasını oku
        df = pd.read_csv(file_path)
        
        # Dosya adından kategori çıkar (uzantıyı kaldır ve '_Articles' kısmını sil)
        category = os.path.splitext(file)[0].replace('_Articles', '').replace('_', ' ')
        
        # Kategori sütunu ekle
        df['Category'] = category
        
        # Birleştir
        combined_df = pd.concat([combined_df, df], ignore_index=True)

# Birleştirilmiş veriyi CSV olarak kaydet
output_path = r'C:\Web_Scraping\PubMed_Cancer_Articles\final_combined_cancer_articles.csv'
combined_df.to_csv(output_path, index=False)

print(f"Birleştirme tamamlandı. Dosya kaydedildi: {output_path}")
import csv
import re

# Dosya yolu
file_path = r"C:\Users\memir\Desktop\Yaz. Lab\Web_Scraping\PubMed_Cancer_Articles\Merged Article CSV Files\balanced_cleaned_articles_final.csv"
output_file = r"C:\Users\memir\Desktop\cleaned_articles_final.csv"  # Tek dosya olarak kaydedilecek

# Metin temizleme fonksiyonu
def clean_text(text):
    if not text:
        return ""
    text = text.lower()  # Küçük harfe çevir
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Noktalama ve özel karakterleri kaldır
    text = re.sub(r'\s+', ' ', text).strip()  # Fazla boşlukları temizle
    return text

# Dosyayı işle
with open(file_path, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    headers = next(reader)  # Sütun başlıklarını al

    # cleaned_title ve cleaned_abstract sütunlarını ekle
    if 'title' in headers and 'abstract' in headers:
        headers.append('cleaned_title')
        headers.append('cleaned_abstract')
    else:
        raise ValueError("Title veya Abstract sütunu bulunamadı.")

    rows = []

    for row in reader:
        title_index = headers.index('title')  # Title sütununun indexini bul
        abstract_index = headers.index('abstract')  # Abstract sütununun indexini bul
        
        # Fazla boşlukları temizle
        row[title_index] = re.sub(r'\s+', ' ', row[title_index]).strip()
        row[abstract_index] = re.sub(r'\s+', ' ', row[abstract_index]).strip()
        
        cleaned_title = clean_text(row[title_index])  # Title'ı temizle
        cleaned_abstract = clean_text(row[abstract_index])  # Abstract'ı temizle
        row.append(cleaned_title)  # Temizlenen title'ı ekle
        row.append(cleaned_abstract)  # Temizlenen abstract'ı ekle
        rows.append(row)

    # Tek dosya olarak kaydet
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"{output_file} kaydedildi.")
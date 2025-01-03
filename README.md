---

## README.md

# 🧠 Cancer Classification with Transformer Models  
Bu proje, PubMed'den elde edilen makale verileriyle 14 farklı kanser türünü sınıflandırmayı hedeflemektedir. Farklı transformer modelleri kullanılarak veriler işlenmiş, sınıflandırılmış ve performansları karşılaştırılmıştır.

---

## 📁 Proje Yapısı  
- **Kodlar:**  
  - `ALBERT_Model.py` – ALBERT modeliyle sınıflandırma  
  - `DistilBERT_Model.py` – DistilBERT modeliyle sınıflandırma  
  - `ELECTRA_Model.py` – ELECTRA modeliyle sınıflandırma  
  - `MiniLM_Model.py` – MiniLM modeliyle sınıflandırma  
  - `TinyBERTModel.py` – TinyBERT modeliyle sınıflandırma  

- **Veri Seti:**  
  Projede 14 farklı kanser türüne ait metin verileri kullanılmıştır. Bu veri setleri, başlık ve özet bilgilerini içeren temizlenmiş CSV dosyalarından oluşmaktadır.  
  
  ### 📂 Kanser Türleri:  
  - AI-Related  
  - Bladder Cancer  
  - Breast Cancer  
  - Colon or Rectal Cancer  
  - Endometrial Cancer  
  - Hodgkin Lymphoma or Non-Hodgkin Lymphoma  
  - Kidney Cancer  
  - Leukemia or Acute Myeloid Leukemia  
  - Liver Cancer  
  - Lung or Pulmonary Cancer  
  - Melanoma or Skin Cancer  
  - Pancreatic Cancer  
  - Prostate Cancer  
  - Thyroid Cancer  

---

## 🚀 Kullanılan Teknolojiler  
- **Dil ve Framework:** Python (PyTorch, Transformers, scikit-learn)  
- **Modeller:**  
  - ALBERT (`albert-base-v2`)  
  - DistilBERT (`distilbert-base-uncased`)  
  - ELECTRA (`electra-base-discriminator`)  
  - MiniLM (`microsoft/MiniLM-L12-H384-uncased`)  
  - TinyBERT (`huawei-noah/TinyBERT_General_4L_312D`)  
- **GPU Desteği:** CUDA (NVIDIA GPU)  
- **Grafik ve Görselleştirme:** Matplotlib  

---

## 📊 Performans Değerlendirmesi  
Modeller, eğitim ve doğrulama aşamalarında şu metriklerle değerlendirilmiştir:  
- **Doğruluk (Accuracy)**  
- **Hassasiyet (Precision)**  
- **Duyarlılık (Recall)**  
- **F1-Skoru (F1-Score)**  
- **ROC-AUC**  
- **Spesifiklik (Specificity)**  

Her model için eğitim kayıpları ve doğrulama kayıpları grafiklerle görselleştirilmiştir. Erken durdurma (early stopping) kullanılarak modelin aşırı öğrenmesi engellenmiştir.

---

## 🛠️ Kurulum  
Projenin çalıştırılabilmesi için aşağıdaki adımları takip edebilirsiniz:  
1. Gerekli kütüphaneleri yükleyin:  
   ```bash
   pip install transformers torch scikit-learn matplotlib pandas
   ```

2. Veri setini Google Drive’a yükleyin ve `file_path` kısmını uygun şekilde değiştirin.  

3. İlgili modeli çalıştırın:  
   ```bash
   python ALBERT_Model.py
   ```

---

## 📥 Veri Seti Erişimi  
Veri seti doğrudan GitHub’a yüklenemeyecek kadar büyük olduğu için **Google Drive** üzerinden erişilebilir.   
**📎 Veri Seti İndirme Linki:** [Drive'dan İndir](https://drive.google.com/drive/folders/15OcbsSSkBsMi5N-xi2re8RsWerMY5SU5)  

---

## 📈 Model Eğitimi ve Çalıştırma  
Kod dosyaları, her model için benzer bir eğitim döngüsü içermektedir. Veriler işlenip modele verilmekte, eğitim sonrası sonuçlar kaydedilmektedir. Aşağıda bir örnek eğitim çıktısı yer almaktadır:  
```bash
Epoch 1/10 - Train Loss: 0.45, Validation Loss: 0.38
Accuracy: 92.3%, Precision: 91.5%, Recall: 89.6%, F1-Score: 90.5%
```

---

import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import torch
import matplotlib.pyplot as plt
import time

# 1. CSV dosyasını oku
file_path = '/content/drive/MyDrive/Colab Notebooks/data/cleaned_articles_final.csv'
df = pd.read_csv(file_path)

# 2. cleaned_title ve cleaned_abstract sütunlarını birleştir
df['combined_text'] = (df['cleaned_title'].fillna('') + " " + df['clean_abstract'].fillna(''))

# 3. Kategorileri encode et
label_encoder = LabelEncoder()
df['category'] = label_encoder.fit_transform(df['category'])

# 4. TinyBERT Tokenizer ve Modeli yükle
tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
model = AutoModelForSequenceClassification.from_pretrained(
    "huawei-noah/TinyBERT_General_4L_312D",
    num_labels=len(df['category'].unique())
)

# 5. GPU kullanımı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 6. Metinleri tokenize et
inputs = tokenizer(
    df['combined_text'].tolist(),
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)
labels = torch.tensor(df['category'].tolist())

# 7. Veriyi eğitim ve test olarak böl
train_inputs, val_inputs, train_labels, val_labels = train_test_split(
    inputs['input_ids'], labels, test_size=0.2, random_state=42, shuffle=True
)

# 8. TensorDataset oluştur
train_inputs = {
    'input_ids': train_inputs,
    'attention_mask': torch.ones_like(train_inputs)
}
val_inputs = {
    'input_ids': val_inputs,
    'attention_mask': torch.ones_like(val_inputs)
}

train_dataset = TensorDataset(
    train_inputs['input_ids'], train_inputs['attention_mask'], train_labels
)
val_dataset = TensorDataset(
    val_inputs['input_ids'], val_inputs['attention_mask'], val_labels
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# 9. Optimizer ve Loss fonksiyonu
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

# 10. Early Stopping
epochs = 10
patience = 2
best_val_loss = float('inf')
counter = 0
training_losses = []
validation_losses = []
start_time = time.time()

# 11. Eğitim Döngüsü
for epoch in range(epochs):
    model.train()
    total_loss = 0
    print(f"\nEpoch {epoch+1}/{epochs} başlıyor...")

    for batch in train_loader:
        b_input_ids, b_attn_mask, b_labels = [item.to(device) for item in batch]

        outputs = model(input_ids=b_input_ids, attention_mask=b_attn_mask, labels=b_labels)
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    training_losses.append(avg_train_loss)

    # Validation
    model.eval()
    val_loss = 0
    all_preds, all_labels, all_logits = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            b_input_ids, b_attn_mask, b_labels = [item.to(device) for item in batch]
            outputs = model(input_ids=b_input_ids, attention_mask=b_attn_mask, labels=b_labels)
            val_loss += outputs.loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(b_labels.cpu().numpy())
            all_logits.append(torch.softmax(outputs.logits, dim=1))

    avg_val_loss = val_loss / len(val_loader)
    validation_losses.append(avg_val_loss)

    # Performans metrikleri
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Specificity ve ROC-AUC
    tn, fp, fn, tp = conf_matrix.ravel() if conf_matrix.size == 4 else (0, 0, 0, 0)
    sensitivity = recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    auc_score = roc_auc_score(all_labels, torch.cat(all_logits).cpu(), multi_class='ovr')

    print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {sensitivity:.4f}, Specificity: {specificity:.4f}, F1-Score: {f1:.4f}, AUC: {auc_score:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model_tinybert.pth')
    else:
        counter += 1
        if counter >= patience:
            print("Erken durdurma aktif. Eğitim durduruluyor.")
            break

end_time = time.time()
print(f"\nToplam Eğitim Süresi: {end_time - start_time:.2f} saniye")

plt.plot(training_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

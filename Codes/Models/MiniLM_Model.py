import pandas as pd
import torch
import time
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import GradScaler, autocast

# 1. CSV dosyasını oku
file_path = '/content/drive/MyDrive/Colab Notebooks/data/cleaned_articles_final.csv'
df = pd.read_csv(file_path)

# 2. Metinleri birleştir
df['combined_text'] = (df['cleaned_title'].fillna('') + " " + df['clean_abstract'].fillna(''))

# 3. Kategorileri encode et
label_encoder = LabelEncoder()
df['category'] = label_encoder.fit_transform(df['category'])

# 4. MiniLM Tokenizer ve Modeli yükle
tokenizer = AutoTokenizer.from_pretrained("microsoft/MiniLM-L12-H384-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/MiniLM-L12-H384-uncased",
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

# 7. Eğitim ve test verilerini ayır
train_input_ids, val_input_ids, train_labels, val_labels = train_test_split(
    inputs['input_ids'], labels, test_size=0.2, random_state=42, shuffle=True
)

train_attention_mask, val_attention_mask = train_test_split(
    inputs['attention_mask'], test_size=0.2, random_state=42, shuffle=True
)

train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
val_dataset = TensorDataset(val_input_ids, val_attention_mask, val_labels)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4, pin_memory=True)

# 8. Optimizer ve Loss Fonksiyonu
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
loss_fn = torch.nn.CrossEntropyLoss()
scaler = GradScaler()

# 9. Eğitim Parametreleri
epochs = 10
patience = 2
best_val_loss = float('inf')
counter = 0
training_losses = []
validation_losses = []
start_time = time.time()

# Specificity hesaplama fonksiyonu
def calculate_specificity(conf_matrix):
    specificity_per_class = []
    for i in range(len(conf_matrix)):
        tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
        fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_per_class.append(specificity)
    return sum(specificity_per_class) / len(specificity_per_class)

# 10. Eğitim Döngüsü
for epoch in range(epochs):
    model.train()
    total_loss = 0
    print(f"\nEpoch {epoch+1}/{epochs} başlıyor...")

    for batch in train_loader:
        b_input_ids, b_attn_mask, b_labels = [item.to(device) for item in batch]

        optimizer.zero_grad()
        with autocast('cuda'):
            outputs = model(input_ids=b_input_ids, attention_mask=b_attn_mask, labels=b_labels)
            loss = outputs.loss
        scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

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
            with autocast('cuda'):
                outputs = model(input_ids=b_input_ids, attention_mask=b_attn_mask, labels=b_labels)
            val_loss += outputs.loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(b_labels.cpu().numpy())
            all_logits.append(torch.softmax(outputs.logits, dim=1))

    avg_val_loss = val_loss / len(val_loader)
    validation_losses.append(avg_val_loss)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    specificity = calculate_specificity(conf_matrix)
    all_logits_tensor = torch.cat(all_logits).cpu()
    all_labels_one_hot = label_binarize(all_labels, classes=list(range(len(df['category'].unique()))))
    auc_score = roc_auc_score(all_labels_one_hot, all_logits_tensor.numpy(), multi_class='ovr')

    print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Specificity: {specificity:.4f}, F1-Score: {f1:.4f}, AUC: {auc_score:.4f}")
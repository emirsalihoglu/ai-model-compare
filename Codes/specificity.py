from sklearn.metrics import confusion_matrix

# 1. Model eğitimi tamamlandıktan sonra validation set üzerinde tahmin yap
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in val_loader:
        b_input_ids, b_attn_mask, b_labels = [item.to(device) for item in batch]
        outputs = model(input_ids=b_input_ids, attention_mask=b_attn_mask)
        
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(b_labels.cpu().numpy())

# 2. Confusion Matrix Hesaplama
conf_matrix = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", conf_matrix)

# 3. Specificity Hesaplama Fonksiyonu
def calculate_specificity(conf_matrix):
    specificity_per_class = []
    for i in range(conf_matrix.shape[0]):
        tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
        fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_per_class.append(specificity)
    return sum(specificity_per_class) / len(specificity_per_class)

# 4. Specificity Hesapla
specificity = calculate_specificity(conf_matrix)
print(f"Specificity: {specificity:.4f}")
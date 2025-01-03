import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

# CSV dosyasini oku
file_path = r"C:\Users\memir\Desktop\Yaz. Lab\Web_Scraping\PubMed_Cancer_Articles\Merged Article CSV Files\cleaned_articles.csv"
df = pd.read_csv(file_path)

# Kategorileri sayısal etiketlere dönüştür
label_encoder = LabelEncoder()
df['category'] = label_encoder.fit_transform(df['category'])

# TF-IDF vektorlestirme
vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(df['clean_abstract'])
y = df['category']

# Veriyi bol (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost Modeli
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Tahmin ve Değerlendirme
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print(f"Doğruluk: {accuracy:.2f}")
print(report)
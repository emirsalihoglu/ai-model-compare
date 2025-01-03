from Bio import Entrez
import os
import re
import time
import csv

# Specify your email address for PubMed API
Entrez.email = "emir.m.salihoglu@gmail.com"  # Enter a valid email address

# Folders to save files
BASE_FOLDER = r"C:\Users\memir\Desktop\Yaz. Lab\Web_Scraping\PubMed_Cancer_Articles"
os.makedirs(BASE_FOLDER, exist_ok=True)

# Function to sanitize invalid characters (for file/folder names)
def sanitize_name(name):
    return re.sub(r'[<>:"/\\|?*]', '', name)

# List of cancer types
CANCER_TYPES = [
    "Colon Cancer OR Rectal Cancer",
    "Leukemia OR Acute Myeloid Leukemia OR Chronic Lymphocytic Leukemia",
    "Liver Cancer",
    "Lung Cancer OR Pulmonary Cancer",
    "Melanoma OR Skin Cancer",
    "Hodgkin Lymphoma OR Non-Hodgkin Lymphoma",
]

# AI-related keywords
AI_KEYWORDS = [
    "machine learning",
    "deep learning",
    "artificial intelligence"
]

# Extract article title
def extract_title(article):
    match = re.search(r"TI  - (.+?)(?:\n[A-Z]{2}  -|\Z)", article, re.DOTALL)
    return match.group(1).replace("\n", " ").strip() if match else "No Title"

# Extract article abstract
def extract_abstract(article):
    match = re.search(r"AB  - (.+?)(?:\n[A-Z]{2}  -|\Z)", article, re.DOTALL)
    return match.group(1).replace("\n", " ").strip() if match else "No Abstract"

# Save article as txt file
def save_article_txt(folder, article, index):
    title = extract_title(article)
    abstract = extract_abstract(article)

    filename = f"Article_{index}.txt"
    filepath = os.path.join(folder, filename)

    with open(filepath, "w", encoding="utf-8") as file:
        file.write(f"Title: {title}\n\n")
        file.write(f"Abstract: {abstract}\n")

    return {"Title": title, "Abstract": abstract}

# Save articles to CSV
def save_to_csv(filepath, articles):
    with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Title", "Abstract"])
        writer.writeheader()
        writer.writerows(articles)

# Fetch cancer articles from PubMed
def fetch_cancer_articles(cancer, total_results=10000, batch_size=200):
    cancer_folder = os.path.join(BASE_FOLDER, sanitize_name(cancer))
    os.makedirs(cancer_folder, exist_ok=True)

    csv_path = os.path.join(cancer_folder, f"{sanitize_name(cancer)}_Articles.csv")
    articles = []

    query = f"{cancer} NOT {' NOT '.join(AI_KEYWORDS)}"

    print(f"Fetching general articles for {cancer}...")
    
    for retstart in range(0, total_results, batch_size):
        handle = Entrez.esearch(db="pubmed", term=query, retmax=batch_size, retstart=retstart)
        record = Entrez.read(handle)
        handle.close()

        if not record["IdList"]:
            print(f"No more articles found for {cancer}.")
            break

        handle = Entrez.efetch(db="pubmed", id=record["IdList"], rettype="medline", retmode="text")
        articles_text = handle.read().split("\n\n")
        handle.close()

        for i, article in enumerate(articles_text):
            article_data = save_article_txt(cancer_folder, article, retstart + i + 1)
            articles.append(article_data)

        print(f"Saved {len(articles)} articles for {cancer}.")
        time.sleep(2)

    save_to_csv(csv_path, articles)

# Fetch AI-related articles for each cancer type
def fetch_cancer_ai_articles(cancer, total_results=10000, batch_size=200):
    cancer_folder = os.path.join(BASE_FOLDER, sanitize_name(cancer))
    os.makedirs(cancer_folder, exist_ok=True)

    csv_path = os.path.join(cancer_folder, f"{sanitize_name(cancer)}_AI_Articles.csv")
    articles = []

    ai_query = f"{cancer} AND ({' OR '.join(AI_KEYWORDS)})"

    print(f"Fetching AI-related articles for {cancer}...")

    for retstart in range(0, total_results, batch_size):
        handle = Entrez.esearch(db="pubmed", term=ai_query, retmax=batch_size, retstart=retstart)
        record = Entrez.read(handle)
        handle.close()

        if not record["IdList"]:
            print(f"No more AI articles found for {cancer}.")
            break

        handle = Entrez.efetch(db="pubmed", id=record["IdList"], rettype="medline", retmode="text")
        articles_text = handle.read().split("\n\n")
        handle.close()

        for i, article in enumerate(articles_text):
            article_data = save_article_txt(cancer_folder, article, retstart + i + 1)
            articles.append(article_data)

        print(f"Saved {len(articles)} AI articles for {cancer}.")
        time.sleep(2)

    save_to_csv(csv_path, articles)

# Fetch articles for each cancer type
for cancer in CANCER_TYPES:
    fetch_cancer_articles(cancer)
    fetch_cancer_ai_articles(cancer)

print("All articles have been successfully saved.")
import pandas as pd
import re
import torch
from transformers import BertModel, BertTokenizer, T5Tokenizer, T5ForConditionalGeneration
from sklearn.cluster import KMeans
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge

# Charger le modèle BioBERT pré-entraîné
model_bio = BertModel.from_pretrained("dmis-lab/biobert-v1.1")
tokenizer_bio = BertTokenizer.from_pretrained("dmis-lab/biobert-v1.1")

# Charger le modèle T5 pré-entraîné
model_t5 = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer_t5 = T5Tokenizer.from_pretrained("t5-small")


def format_function(data):
    if isinstance(data, str):
        # Remplacer les citations par la balise <cit>
        data = re.sub(r'\[\d+[,0-9/-]*\]', r' <cit> ', data)
        data = re.sub(r'\[\d+[" ,"/0-9/-]*\]', r' <cit> ', data)

        # Supprimer le texte entre parenthèses et crochets
        data = re.sub(r'\([^)]+\)', '', data)
        data = re.sub(r'\[.*?\]', '', data)

        # Remplacer les chiffres par la balise <dig> uniquement pour les chaînes de caractères
        data = re.sub(r'\b\d+\b', ' <dig> ', data)

        # Supprimer les tables et les figures
        data = re.sub(r'\ntable \d+.*?\n', '', data)
        data = re.sub(r'.*\t.*?\n', '', data)
        data = re.sub(r'\nfigure \d+.*?\n', '', data)
        data = re.sub(r'[(]figure \d+.*?[)]', '', data)
        data = re.sub(r'[(]fig. \d+.*?[)]', '', data)
        data = re.sub(r'[(]fig \d+.*?[)]', '', data) 

    return data

# Lire les 15000 premiers textes de la colonne 'cleaned_text' à partir du fichier CSV
file_path = '/kaggle/input/pfe-1-1/output1.csv'
data = pd.read_csv(file_path, usecols=['cleaned_text1', 'shorter_abstract']).head(15000)

# Supprimer les lignes contenant des valeurs nulles dans la colonne 'cleaned_text'
data.dropna(subset=['cleaned_text1', 'shorter_abstract'], inplace=True)

# Appliquer la fonction de formatage à chaque ligne de texte
data['texte_formate'] = data['cleaned_text1'].apply(format_function)

# Fonction pour générer des représentations vectorielles avec BioBERT
def generate_embeddings_bio(text):
    inputs = tokenizer_bio(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model_bio(**inputs)
    hidden_states = outputs.last_hidden_state
    embeddings = hidden_states.mean(dim=1)
    return embeddings

# Appliquer la fonction de génération de représentations vectorielles à chaque texte prétraité
data['embeddings'] = data['texte_formate'].apply(generate_embeddings_bio)

# Utiliser les embeddings pour regrouper les phrases en clusters avec K-Means
X = torch.cat(data['embeddings'].tolist(), dim=0).numpy()

# Définir le nombre de clusters
n_clusters = 1  # Ou tout autre nombre de clusters que vous souhaitez former

# Utiliser K-Means pour regrouper les embeddings en clusters
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(X)

# Sélectionner un représentant unique de chaque cluster comme résumé
summary_indices = []
for i in range(n_clusters):
    cluster_indices = np.where(kmeans.labels_ == i)[0]
    cluster_center = kmeans.cluster_centers_[i]
    distances_to_center = np.linalg.norm(X[cluster_indices] - cluster_center, axis=1)
    summary_index = cluster_indices[np.argmin(distances_to_center)]
    summary_indices.append(summary_index)

# Stocker les résumés dans une liste
generated_summaries = []
reference_summaries = data.iloc[summary_indices]['shorter_abstract'].tolist()

# Utiliser T5 pour générer le résumé abstrait
for idx in summary_indices:
    source_text = data.iloc[idx]['texte_formate']
    inputs = tokenizer_t5(source_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model_t5.generate(inputs.input_ids, num_beams=4, max_length=150, early_stopping=True)
    summary_text = tokenizer_t5.decode(summary_ids[0], skip_special_tokens=True)
    generated_summaries.append(summary_text)

# Calculer les scores BLEU et ROUGE
bleu_scores = corpus_bleu([[ref.split()] for ref in reference_summaries], [gen.split() for gen in generated_summaries])
rouge = Rouge()
rouge_scores = rouge.get_scores(generated_summaries, reference_summaries, avg=True)

# Afficher les scores BLEU et ROUGE
print("Score BLEU:", bleu_scores)
print("Score ROUGE:", rouge_scores)

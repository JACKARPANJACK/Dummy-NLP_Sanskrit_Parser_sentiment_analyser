import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math

with open('python experiment/.venv/Scripts/verse.json', encoding='utf-8') as f:
    verses = json.load(f)
# Load translation.json to inject translations before building corpus
try:
    with open('python experiment/.venv/Scripts/translation.json', encoding='utf-8') as tf:
        t_data = json.load(tf)
    trans_map = {}
    for t in t_data:
        if t.get("lang", "").lower() == "english":
            vid = t.get("verse_id")
            if vid not in trans_map:
                trans_map[vid] = []
            trans_map[vid].append(t.get("description", "").strip())
    
    for v in verses:
        vid = v.get("id")
        if vid in trans_map:
            v["translation"] = "\n\n".join(trans_map[vid])
except Exception as e:
    print("Warning: could not load translations", e)
corpus = []
for v in verses:
    text_parts = []
    if v.get("translation"):
        text_parts.append(v["translation"])
    if v.get("word_meanings"):
        text_parts.append(v["word_meanings"])
    if v.get("transliteration"):
        text_parts.append(v["transliteration"])
    corpus.append(" ".join(text_parts))

print("Corpus size:", len(corpus))
tf_vectorizer = TfidfVectorizer(stop_words="english")
tf_matrix = tf_vectorizer.fit_transform(corpus)

query = "peace"
q_vec = tf_vectorizer.transform([query.lower()])
similarities = cosine_similarity(q_vec, tf_matrix)[0]
idxs = similarities.argsort()[::-1][:5]
for i in idxs:
    print(i, similarities[i], verses[i].get("word_meanings")[:50])
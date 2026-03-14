import re
import json
import unicodedata
import os

import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
import nltk

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from collections import Counter
from typing import List, Dict


# ---------------------------
# NLTK Setup
# ---------------------------
for _pkg, _name in [
    ("sentiment/vader_lexicon.zip", "vader_lexicon"),
    ("corpora/stopwords.zip",       "stopwords"),
    ("corpora/wordnet.zip",         "wordnet"),
    ("corpora/omw-1.4.zip",         "omw-1.4"),
    ("tokenizers/punkt.zip",        "punkt"),
]:
    try:
        nltk.data.find(_pkg)
    except LookupError:
        nltk.download(_name, quiet=True)
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()

try:
    ENGLISH_STOPWORDS = set(stopwords.words("english"))
except:
    ENGLISH_STOPWORDS = set()


# ---------------------------
# Unicode Helpers
# ---------------------------

def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


DEV_RE = r"\u0900-\u097F"
ENGLISH_WORD_RE = re.compile(r"[A-Za-z]+")

# ---------------------------
# Preprocessing (Lowercase -> Punctuation removal -> Tokenization -> Stopword removal -> Lemmatization)
# ---------------------------
def lowercase(text: str) -> str:
    return (text or "").lower()
def remove_punctuation(text: str) -> str:
    return re.sub(rf"[^\w\s{DEV_RE}]", " ", text or "")
def remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in ENGLISH_STOPWORDS and len(t) > 1]
def lemmatize(tokens: List[str]) -> List[str]:
    return [lemmatizer.lemmatize(t) for t in tokens]
#Fixed
def process(text: str) -> List[str]:
    text = lowercase(text)
    text = remove_punctuation(text)
    tokens, _ = tokenize(text, strip_diac=True, ascii_map=True)  # unpack tuple
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return [t for t in tokens if t.strip()]
# ---------------------------
# Tokenizer
# ---------------------------

def tokenize(text):

    text = normalize_text(text)

    tokens = []
    types = []

    i = 0

    while i < len(text):

        if text[i].isspace():
            i += 1
            continue

        m = NUMBER_RE.match(text, i)

        if m:
            tokens.append(m.group())
            types.append("NUMBER")
            i = m.end()
            continue

        m = WORD_RE.match(text, i)

        if m:

            tok = m.group()

            if re.search(DEV_RE, tok):
                types.append("DEVANAGARI_WORD")
            else:
                types.append("WORD")

            tokens.append(tok)
            i = m.end()
            continue

        m = PUNCT_RE.match(text, i)

        if m:
            tokens.append(m.group())
            types.append("PUNCT")
            i = m.end()
            continue

        tokens.append(text[i])
        types.append("UNKNOWN")
        i += 1

    return tokens, types


# ---------------------------
# Safe NLP Pipeline
# ---------------------------

def lowercase(text):
    return text.lower()


def remove_punctuation(text):
    return re.sub(rf"[^\w\s{DEV_RE}]", " ", text)


def remove_stopwords(tokens):

    cleaned = []

    for t in tokens:

        if ENGLISH_WORD_RE.fullmatch(t):

            if t not in ENGLISH_STOPWORDS:
                cleaned.append(t)

        else:
            cleaned.append(t)

    return cleaned


def lemmatize(tokens):

    out = []

    for t in tokens:

        try:

            if ENGLISH_WORD_RE.fullmatch(t):
                out.append(lemmatizer.lemmatize(t))

            else:
                out.append(t)

        except:
            out.append(t)

    return out


def process(text):

    try:

        text = normalize_text(text)
        text = lowercase(text)
        text = remove_punctuation(text)

        tokens, _ = tokenize(text)

        tokens = [t.strip() for t in tokens if t.strip()]

        tokens = remove_stopwords(tokens)

        tokens = lemmatize(tokens)

        return tokens

    except:
        return []


# ---------------------------
# Token Statistics
# ---------------------------

def token_stats(tokens, types):

    stats = Counter(types)

    return {

        "TOTAL": len(tokens),
        "WORD": stats.get("WORD", 0),
        "DEVANAGARI_WORD": stats.get("DEVANAGARI_WORD", 0),
        "NUMBER": stats.get("NUMBER", 0),
        "PUNCT": stats.get("PUNCT", 0)
    }


# ---------------------------
# Sentiment
# ---------------------------

def sentiment_label(text):

    words = re.findall(r"[A-Za-z]+", text)

    if not words:
        return "Neutral (Non-English Input)", {"compound": 0}

    scores = sia.polarity_scores(" ".join(words))

    c = scores["compound"]

    if c >= 0.05:
        label = "Positive"
    elif c <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return label, scores


# ---------------------------
# Recommendation Engine
# ---------------------------
def recommend_verses_for_input(input_text: str, verses: List[Dict], top_n: int=5,
                               strip_diac: bool=False, ascii_map: bool=False) -> List[Tuple[Dict, float]]:
    """
    1. Analyze input -> sentiment label and english words
    2. Filter verses that have the same sentiment label (by precomputed sentiment in verse['sentiment_label'])
       if none, fallback to whole dataset
    3. Score by overlap of English words between input and verse.word_meanings/transliteration
    Returns list of (verse, score) sorted desc
    """
    # tokenize input & extract english words
    tokens, types = process(input_text)
    eng_words = [t.lower() for t in tokens if ENGLISH_WORD_RE.fullmatch(t)]
    eng_text = " ".join(eng_words)
    input_label, _ = sentiment_label_and_scores_from_text_english(eng_text)

def recommend_verses(input_text, verses):

    tokens = process(input_text)

    input_set = set(tokens)

    scored = []

    for v in verses:

        try:

            verse_tokens = process(
                v.get("word_meanings") or
                v.get("transliteration") or
                v.get("text") or ""
            )

        except:
            verse_tokens = []

        verse_set = set(verse_tokens)

        common = input_set.intersection(verse_set)

        if len(verse_set) == 0:
            score = 0
        else:
            score = len(common) / len(verse_set)

        scored.append((v, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:6]


# ---------------------------
# UI Application
# ---------------------------

class GitaApp:

    def __init__(self, root):

        self.root = root
        root.title("Gita Wisdom Engine")

        self.verses = []

        tk.Label(root, text="Enter Text").pack()

        self.input_box = scrolledtext.ScrolledText(root, height=5)
        self.input_box.pack(fill="both")

        btn_frame = tk.Frame(root)
        btn_frame.pack()

        tk.Button(btn_frame, text="Analyze", command=self.analyze).pack(side="left")
        tk.Button(btn_frame, text="Load JSON", command=self.load_verses).pack(side="left")
        tk.Button(btn_frame, text="Recommend", command=self.recommend).pack(side="left")

        tk.Label(root, text="Tokens").pack()

        self.token_box = scrolledtext.ScrolledText(root, height=6)
        self.token_box.pack(fill="both")

        tk.Label(root, text="Statistics").pack()

        self.stat_box = scrolledtext.ScrolledText(root, height=4)
        self.stat_box.pack(fill="both")

        tk.Label(root, text="Sentiment").pack()

        self.sentiment_box = scrolledtext.ScrolledText(root, height=3)
        self.sentiment_box.pack(fill="both")

        tk.Label(root, text="Recommendations").pack()

        self.output_box = scrolledtext.ScrolledText(root, height=10)
        self.output_box.pack(fill="both")


    def load_verses(self):

        path = filedialog.askopenfilename()

        if not path:
            return

        with open(path, encoding="utf-8") as f:
            self.verses = json.load(f)

        messagebox.showinfo("Loaded", f"{len(self.verses)} verses loaded")


    def analyze(self):

        text = self.input_box.get("1.0", tk.END)

        tokens, types = tokenize(text)

    def run_analysis(self):
        text = self.input_box.get("1.0", tk.END).strip()
        strip = self.strip_var.get()
        ascii_map = self.ascii_var.get()

        # Use tokenize() directly for display (gives tokens + types)
        tokens, types = tokenize(text, strip_diac=strip, ascii_map=ascii_map)
        stats = token_stats(tokens, types)

        # Use process() only for NLP (clean tokens for sentiment)
        clean_tokens = process(text)
        eng_text = " ".join(t for t in clean_tokens if ENGLISH_WORD_RE.fullmatch(t))
        label, scores = sentiment_label_and_scores_from_text_english(eng_text)

        self.token_box.delete("1.0", tk.END)
        self.token_box.insert(tk.END,
            "\n".join(f"{t} -> {ty}" for t, ty in zip(tokens, types))
        )

        self.stat_box.delete("1.0", tk.END)

        for k, v in stats.items():
            self.stat_box.insert(tk.END, f"{k}: {v}\n")

        self.sentiment_box.delete("1.0", tk.END)
        self.sentiment_box.insert(tk.END, f"{label}\nScores:{scores}")


    def recommend(self):

        try:

            text = self.input_box.get("1.0", tk.END)

            if not self.verses:
                messagebox.showwarning("No verses", "Load Gita JSON first")
                return

            results = recommend_verses(text, self.verses)

            self.output_box.delete("1.0", tk.END)

            for v, score in results:

                self.output_box.insert(
                    tk.END,
                    f"Chapter {v['chapter_number']} Verse {v['verse_number']} (score {score:.2f})\n"
                )

                preview = (v.get("text") or "").splitlines()[0]

                self.output_box.insert(tk.END, f"{preview}\n\n")

        except Exception as e:

            messagebox.showerror("Error", str(e))


# ---------------------------
# Run
# ---------------------------

if __name__ == "__main__":

    root = tk.Tk()

    app = GitaApp(root)

    root.mainloop()
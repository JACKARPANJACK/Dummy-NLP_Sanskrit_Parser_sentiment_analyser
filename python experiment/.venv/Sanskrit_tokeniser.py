import re
import unicodedata
import tkinter as tk
from tkinter import scrolledtext

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize sentiment analyzer
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()

# --- Unicode helpers ---

def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text)

def strip_diacritics(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))

def iast_to_ascii(text: str) -> str:
    mapping = {
        "ā":"a","ī":"i","ū":"u","ṛ":"r","ṝ":"r","ḷ":"l",
        "ṅ":"n","ñ":"n","ṇ":"n","ṣ":"s","ś":"s","ḥ":"h","ṁ":"m",
        "Ā":"A","Ī":"I","Ū":"U","Ṛ":"R","Ṝ":"R","Ḷ":"L",
        "Ṅ":"N","Ñ":"N","Ṇ":"N","Ṣ":"S","Ś":"S","Ḥ":"H","Ṁ":"M"
    }
    return "".join(mapping.get(ch, ch) for ch in text)

# --- Tokenizer ---

WORD_RE = re.compile(r"[A-Za-z\u0100-\u024F\u1E00-\u1EFF]+(?:[-'][A-Za-z]+)*")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
PUNCT_RE = re.compile(r"[^\s\w]")

def tokenize(text, strip_diac=False, ascii_map=False):
    text = normalize_text(text)

    if strip_diac:
        text = strip_diacritics(text)

    if ascii_map:
        text = iast_to_ascii(text)

    tokens = []
    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue

        m = NUMBER_RE.match(text, i)
        if m:
            tokens.append(m.group(0))
            i = m.end()
            continue

        m = WORD_RE.match(text, i)
        if m:
            tokens.append(m.group(0))
            i = m.end()
            continue

        m = PUNCT_RE.match(text, i)
        if m:
            tokens.append(m.group(0))
            i = m.end()
            continue

        tokens.append(text[i])
        i += 1

    return tokens

# --- Sentiment ---

def english_sentiment(tokens):
    english_words = [t for t in tokens if re.fullmatch(r"[A-Za-z]+", t)]
    if not english_words:
        return "Neutral (no English detected)"

    text = " ".join(english_words)
    scores = sia.polarity_scores(text)

    if scores["compound"] >= 0.05:
        label = "Positive"
    elif scores["compound"] <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return f"{label}\nScores: {scores}"

# --- Tkinter UI ---

def run_analysis():
    text = input_box.get("1.0", tk.END).strip()

    strip_diac = strip_var.get()
    ascii_map = ascii_var.get()

    tokens = tokenize(text, strip_diac, ascii_map)
    sentiment = english_sentiment(tokens)

    token_box.delete("1.0", tk.END)
    token_box.insert(tk.END, " ".join(tokens))

    sentiment_box.delete("1.0", tk.END)
    sentiment_box.insert(tk.END, sentiment)


root = tk.Tk()
root.title("English + Sanskrit NLP Tokenizer")

# Input
tk.Label(root, text="Enter Text:").pack()
input_box = scrolledtext.ScrolledText(root, height=5)
input_box.pack(fill="both", padx=5, pady=5)

# Options
strip_var = tk.BooleanVar()
ascii_var = tk.BooleanVar()

tk.Checkbutton(root, text="Strip Diacritics", variable=strip_var).pack(anchor="w")
tk.Checkbutton(root, text="Convert IAST → ASCII", variable=ascii_var).pack(anchor="w")

# Run button
tk.Button(root, text="Analyze", command=run_analysis).pack(pady=10)

# Output tokens
tk.Label(root, text="Tokens:").pack()
token_box = scrolledtext.ScrolledText(root, height=3)
token_box.pack(fill="both", padx=5, pady=5)

# Output sentiment
tk.Label(root, text="Sentiment:").pack()
sentiment_box = scrolledtext.ScrolledText(root, height=5)
sentiment_box.pack(fill="both", padx=5, pady=5)

root.mainloop()

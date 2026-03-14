import re
import json
import unicodedata
import tkinter as tk
from tkinter import scrolledtext, filedialog
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------
# NLTK setup
# -------------------------

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except:
    nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()


# -------------------------
# Vedic UI Colors
# -------------------------

BG_COLOR = "#f5e6c8"
TEXT_COLOR = "#3b2b1a"
FRAME_COLOR = "#c48a2c"
GOLD = "#d4af37"


# -------------------------
# Unicode helpers
# -------------------------

def normalize_text(text):
    return unicodedata.normalize("NFC", text)


DEV_RE = r"\u0900-\u097F"

WORD_RE = re.compile(rf"[A-Za-z{DEV_RE}]+")

NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")

PUNCT_RE = re.compile(r"[^\s\w]")


# -------------------------
# Tokenizer
# -------------------------

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


# -------------------------
# Token statistics
# -------------------------

def token_stats(tokens, types):

    return {
        "TOTAL": len(tokens),
        "WORD": types.count("WORD"),
        "DEVANAGARI": types.count("DEVANAGARI_WORD"),
        "NUMBER": types.count("NUMBER"),
        "PUNCT": types.count("PUNCT")
    }


# -------------------------
# Sentiment
# -------------------------

def sentiment(text):

    words = re.findall(r"[A-Za-z]+", text)

    if not words:
        return "Neutral"

    score = sia.polarity_scores(" ".join(words))

    c = score["compound"]

    if c > 0.05:
        return "Positive"

    elif c < -0.05:
        return "Negative"

    return "Neutral"


# -------------------------
# Mood detection
# -------------------------

MOOD_KEYWORDS = {

"fear":["fear","afraid","panic","anxious"],

"courage":["fight","battle","warrior","strength"],

"dharma":["duty","responsibility","justice"],

"detachment":["loss","attachment","desire"],

"devotion":["love","faith","god","krishna"],

"wisdom":["truth","knowledge","learn"],

"peace":["calm","peace","meditation"]
}


def detect_mood(text):

    text = text.lower()

    for mood,words in MOOD_KEYWORDS.items():

        for w in words:
            if w in text:
                return mood

    return "wisdom"


# -------------------------
# TF-IDF verse search
# -------------------------

verses = []
vectorizer = None
matrix = None


def build_tfidf(verses):

    corpus = []

    for v in verses:

        text = ""

        if v.get("word_meanings"):
            text += v["word_meanings"]

        if v.get("transliteration"):
            text += " " + v["transliteration"]

        corpus.append(text.lower())

    vectorizer = TfidfVectorizer(stop_words="english")

    matrix = vectorizer.fit_transform(corpus)

    return vectorizer, matrix


def recommend_tfidf(query):

    q = vectorizer.transform([query.lower()])

    sim = cosine_similarity(q, matrix).flatten()

    idx = sim.argsort()[::-1][:5]

    return [(verses[i], sim[i]) for i in idx]


# -------------------------
# Reader Page
# -------------------------

def open_reader(verse):

    reader = tk.Toplevel(root)

    reader.title("Gita Reader")

    reader.configure(bg=BG_COLOR)

    header = tk.Label(
        reader,
        text=f"Chapter {verse['chapter_number']} Verse {verse['verse_number']}",
        font=("Noto Serif Devanagari",18,"bold"),
        bg=BG_COLOR,
        fg=GOLD
    )

    header.pack(pady=10)

    text_area = scrolledtext.ScrolledText(
        reader,
        wrap=tk.WORD,
        font=("Noto Serif Devanagari",14),
        bg="#fffaf0",
        fg=TEXT_COLOR
    )

    text_area.pack(fill="both", expand=True, padx=10, pady=10)

    text_area.insert(tk.END,"🕉 Sanskrit Verse\n\n")
    text_area.insert(tk.END, verse.get("text","")+"\n\n")

    if verse.get("transliteration"):
        text_area.insert(tk.END,"🔤 Transliteration\n\n")
        text_area.insert(tk.END,verse["transliteration"]+"\n\n")

    if verse.get("word_meanings"):
        text_area.insert(tk.END,"📖 Word Meanings\n\n")
        text_area.insert(tk.END,verse["word_meanings"]+"\n\n")

    if verse.get("translation"):
        text_area.insert(tk.END,"🌍 Translation\n\n")
        text_area.insert(tk.END,verse["translation"]+"\n\n")

    text_area.insert(tk.END,"\nॐ तत् सत्")

    text_area.config(state="disabled")


# -------------------------
# Load verses
# -------------------------

def load_verses():

    global verses,vectorizer,matrix

    file = filedialog.askopenfilename()

    with open(file,encoding="utf8") as f:

        verses = json.load(f)

    vectorizer,matrix = build_tfidf(verses)

    verse_list.delete(0,tk.END)

    for v in verses:

        verse_list.insert(
        tk.END,
        f"{v['chapter_number']}:{v['verse_number']}"
        )


# -------------------------
# Analyze input
# -------------------------

def analyze():

    text = input_box.get("1.0",tk.END)

    tokens,types = tokenize(text)

    stats = token_stats(tokens,types)

    sent = sentiment(text)

    mood = detect_mood(text)

    token_box.delete("1.0",tk.END)

    token_box.insert(
    tk.END,
    "\n".join(f"{t} -> {ty}" for t,ty in zip(tokens,types))
    )

    stat_box.delete("1.0",tk.END)

    for k,v in stats.items():

        stat_box.insert(tk.END,f"{k}:{v}\n")

    sentiment_box.delete("1.0",tk.END)

    sentiment_box.insert(
    tk.END,
    f"Sentiment: {sent}\nMood: {mood}"
    )


# -------------------------
# Recommendation
# -------------------------

recommended = []


def recommend():

    global recommended

    text = input_box.get("1.0",tk.END)

    results = recommend_tfidf(text)

    recommended = [v for v,_ in results]

    output_box.delete("1.0",tk.END)

    for i,(v,score) in enumerate(results):

        output_box.insert(
        tk.END,
        f"[{i}] Chapter {v['chapter_number']} Verse {v['verse_number']}\n"
        )

    output_box.insert(
    tk.END,
    "\nDouble-click a verse index to open reader"
    )


# -------------------------
# Click handler
# -------------------------

def open_selected_verse(event):

    try:

        line = output_box.get(
        "insert linestart",
        "insert lineend"
        )

        idx = int(line.split("]")[0][1:])

        verse = recommended[idx]

        open_reader(verse)

    except:
        pass


# -------------------------
# UI
# -------------------------

root = tk.Tk()

root.title("Bhagavad Gita Wisdom Engine")

root.configure(bg=BG_COLOR)


header = tk.Label(
root,
text="ॐ तत् सत्\nBhagavad Gita Wisdom Engine",
font=("Noto Serif Devanagari",20,"bold"),
bg=BG_COLOR,
fg=GOLD
)

header.pack(pady=10)


tk.Label(root,
text="Enter thoughts:",
bg=BG_COLOR,
fg=TEXT_COLOR).pack()


input_box = scrolledtext.ScrolledText(
root,
height=5,
bg="#fffaf0"
)

input_box.pack(fill="both",padx=5,pady=5)


btn_frame = tk.Frame(root,bg=BG_COLOR)

btn_frame.pack()


tk.Button(
btn_frame,
text="Analyze",
bg=FRAME_COLOR,
fg="white",
command=analyze
).pack(side="left",padx=5)


tk.Button(
btn_frame,
text="Load Gita JSON",
bg=FRAME_COLOR,
fg="white",
command=load_verses
).pack(side="left",padx=5)


tk.Button(
btn_frame,
text="Recommend Verse",
bg=FRAME_COLOR,
fg="white",
command=recommend
).pack(side="left",padx=5)


tk.Label(root,text="Tokens",bg=BG_COLOR).pack()

token_box = scrolledtext.ScrolledText(root,height=6)

token_box.pack(fill="both")


tk.Label(root,text="Statistics",bg=BG_COLOR).pack()

stat_box = scrolledtext.ScrolledText(root,height=4)

stat_box.pack(fill="both")


tk.Label(root,text="Sentiment / Mood",bg=BG_COLOR).pack()

sentiment_box = scrolledtext.ScrolledText(root,height=3)

sentiment_box.pack(fill="both")


tk.Label(root,text="Recommended Verses",bg=BG_COLOR).pack()

output_box = scrolledtext.ScrolledText(root,height=10)

output_box.pack(fill="both",padx=5,pady=5)

output_box.bind("<Double-Button-1>",open_selected_verse)


verse_list = tk.Listbox(root)

verse_list.pack(fill="both")


root.mainloop()
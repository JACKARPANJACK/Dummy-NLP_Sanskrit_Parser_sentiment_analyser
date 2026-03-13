# gita_recommender.py
import re
import json
import unicodedata
import os
os.environ['TCL_LIBRARY'] = r'C:\Users\KIIT0001\AppData\Local\Programs\Python\Python313\tcl\tcl8.6'
os.environ['TK_LIBRARY'] = r'C:\Users\KIIT0001\AppData\Local\Programs\Python\Python313\tcl\tk8.6'

import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox, simpledialog
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
from typing import Tuple, List, Dict
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------
# NLTK / VADER Setup
# ---------------------------
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
try:
    ENGLISH_STOPWORDS = set(stopwords.words("english"))
except Exception:
    ENGLISH_STOPWORDS = set()
# ---------------------------
# Unicode helpers (as before)
# ---------------------------
def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")

def strip_diacritics(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))

def iast_to_ascii(text: str) -> str:
    mapping = {
        "ā":"a","ī":"i","ū":"u","ṛ":"r","ṝ":"r","ḷ":"l",
        "ṅ":"n","ñ":"n","ṇ":"n","ṣ":"s","ś":"s","ḥ":"h","ṁ":"m",
        "Ā":"A","Ī":"I","Ū":"U","Ṛ":"R","Ṝ":"R","Ḷ":"L",
        "Ṅ":"N","Ñ":"N","Ṇ":"N","Ṣ":"S","Ś":"S","Ḥ":"H","Ṁ":"M"
    }
    return "".join(mapping.get(ch, ch) for ch in text or "")

# ---------------------------
# Regex definitions
# ---------------------------
DEV_RE = r"\u0900-\u097F"
WORD_RE = re.compile(rf"[A-Za-z\u0100-\u024F\u1E00-\u1EFF{DEV_RE}]+(?:[-'][A-Za-z{DEV_RE}]+)*")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
PUNCT_RE = re.compile(r"[^\s\w]")
DEVANAGARI_DETECT = re.compile(rf"[{DEV_RE}]")
ENGLISH_WORD_RE = re.compile(r"[A-Za-z]+")

# ---------------------------
# Preprocessing (Lowercase -> Punctuation removal -> Tokenization -> Stopword removal -> Lemmatization)
# ---------------------------
class DataPreprocessing:
    @staticmethod
    def lowercase(text: str) -> str:
        return (text or "").lower()
    @staticmethod
    def remove_punctuation(text: str) -> str:
        return re.sub(rf"[^\w\s{DEV_RE}]", " ", text or "")
    @staticmethod
    def basic_tokenize(text: str) -> List[str]:
        return [t for t in text.split() if t]
    @staticmethod
    def remove_stopwords(tokens: List[str]) -> List[str]:
        return [t for t in tokens if t not in ENGLISH_STOPWORDS and len(t) > 1]
    @staticmethod
    def lemmatize(tokens: List[str]) -> List[str]:
        return [lemmatizer.lemmatize(t) for t in tokens]
    @staticmethod
    def handle_devanagari(text: str) -> str:
        if DEVANAGARI_DETECT.search(text or ""):
            romanized = devanagari_to_roman(text)
            return iast_to_ascii(strip_diacritics(romanized))
        return text
    def process(self, text: str, is_devanagari_aware: bool = True) -> List[str]:
        if is_devanagari_aware and DEVANAGARI_DETECT.search(text or ""):
            text = self.handle_devanagari(text)
        text = self.lowercase(text)
        text = self.remove_punctuation(text)
        tokens = self.basic_tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return [t for t in tokens if t.strip()]

preprocessor = DataPreprocessing()
# ---------------------------
# Tokenizer (returns tokens and types)
# ---------------------------
def tokenize(text: str, strip_diac: bool=False, ascii_map: bool=False) -> Tuple[List[str], List[str]]:
    text = normalize_text(text)
    if strip_diac:
        text = strip_diacritics(text)
    if ascii_map:
        text = iast_to_ascii(text)

    tokens = []
    token_types = []
    i = 0
    while i < len(text):
        c = text[i]
        if c.isspace():
            i += 1
            continue

        m = NUMBER_RE.match(text, i)
        if m:
            tokens.append(m.group(0)); token_types.append("NUMBER"); i = m.end(); continue

        m = WORD_RE.match(text, i)
        if m:
            token = m.group(0)
            tokens.append(token)
            if DEVANAGARI_DETECT.search(token):
                token_types.append("DEVANAGARI_WORD")
            else:
                token_types.append("WORD")
            i = m.end()
            continue

        m = PUNCT_RE.match(text, i)
        if m:
            tokens.append(m.group(0)); token_types.append("PUNCT"); i = m.end(); continue

        tokens.append(c); token_types.append("UNKNOWN"); i += 1

    return tokens, token_types

# ---------------------------
# Token statistics (counts & type counts)
# ---------------------------
def token_stats(tokens: List[str], types: List[str]) -> Dict[str,int]:
    stats = Counter(types)
    stats_full = {
        "TOTAL": len(tokens),
        "WORD": stats.get("WORD", 0),
        "DEVANAGARI_WORD": stats.get("DEVANAGARI_WORD", 0),
        "NUMBER": stats.get("NUMBER", 0),
        "PUNCT": stats.get("PUNCT", 0),
        "UNKNOWN": stats.get("UNKNOWN", 0)
    }
    return stats_full

# ---------------------------
# Sentiment helpers
# - returns label and full scores dict
# ---------------------------
def sentiment_label_and_scores_from_text_english(text: str) -> Tuple[str, Dict]:
    """
    Use VADER on English text (expect plain ascii). Returns (label, scores).
    """
    if not text or not ENGLISH_WORD_RE.search(text):
        return "Neutral (no English detected)", {"compound": 0.0}

    scores = sia.polarity_scores(text)
    comp = scores.get("compound", 0.0)
    if comp >= 0.05: label = "Positive"
    elif comp <= -0.05: label = "Negative"
    else: label = "Neutral"
    return label, scores

def compute_verse_sentiment(verse: Dict) -> Tuple[str, Dict]:
    """
    Compute sentiment for a verse. Prefer `word_meanings` (English).
    If that's missing, use `transliteration` after ASCII mapping.
    """
    if verse.get("word_meanings"):
        # join english content and run vader
        # strip non-letters so Vader gets plain English words
        text = " ".join(re.findall(r"[A-Za-z']+", verse["word_meanings"]))
        return sentiment_label_and_scores_from_text_english(text)
    elif verse.get("transliteration"):
        # convert diacritics to ascii and feed VADER (not perfect but better than nothing)
        t = iast_to_ascii(strip_diacritics(verse["transliteration"]))
        return sentiment_label_and_scores_from_text_english(" ".join(re.findall(r"[A-Za-z']+", t)))
    else:
        return "Neutral (no English content)", {"compound": 0.0}

# ---------------------------
# Recommendation logic
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
    tokens, types = tokenize(input_text, strip_diac=strip_diac, ascii_map=ascii_map)
    eng_words = [t.lower() for t in tokens if ENGLISH_WORD_RE.fullmatch(t)]
    eng_text = " ".join(eng_words)
    input_label, _ = sentiment_label_and_scores_from_text_english(eng_text)

    # prepare verse candidates with sentiment computed if not present
    candidates = []
    for v in verses:
        # ensure sentiments are computed
        if 'sentiment_label' not in v or 'sentiment_scores' not in v:
            lbl, scores = compute_verse_sentiment(v)
            v['sentiment_label'] = lbl
            v['sentiment_scores'] = scores

    # filter by label
    filtered = [v for v in verses if v.get('sentiment_label') == input_label]
    if not filtered:
        filtered = verses[:]  # fallback

    # scoring: overlap between english words sets (input) and verse's english tokens (word_meanings or transliteration)
    input_set = set(eng_words)
    scored = []
    for v in filtered:
        # get verse english tokens
        if v.get('word_meanings'):
            verse_eng = [w.lower() for w in re.findall(r"[A-Za-z']+", v['word_meanings'])]
        elif v.get('transliteration'):
            t = iast_to_ascii(strip_diacritics(v['transliteration']))
            verse_eng = [w.lower() for w in re.findall(r"[A-Za-z']+", t)]
        else:
            verse_eng = []

        verse_set = set(verse_eng)
        # simple overlap score
        overlap = len(input_set & verse_set)
        # normalize by verse length to prefer concise matches
        denom = max(1, len(verse_set))
        score = overlap / denom
        # small boost based on same sentiment (should be already filtered)
        if v.get('sentiment_label') == input_label:
            score += 0.1
        scored.append((v, score))

    # sort descending by score and return top_n
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]

# ---------------------------
# Tkinter UI
# ---------------------------
class GitaApp:
    def __init__(self, root):
        self.root = root
        root.title("English + Devanagari NLP Tokenizer + Gita Recommender")

        # Input region
        tk.Label(root, text="Enter Text:").pack(anchor="w")
        self.input_box = scrolledtext.ScrolledText(root, height=5)
        self.input_box.pack(fill="both", padx=5, pady=4)

        # options
        self.strip_var = tk.BooleanVar(value=False)
        self.ascii_var = tk.BooleanVar(value=False)
        tk.Checkbutton(root, text="Strip Diacritics", variable=self.strip_var).pack(anchor="w")
        tk.Checkbutton(root, text="Convert IAST → ASCII", variable=self.ascii_var).pack(anchor="w")

        # action buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(fill="x", pady=6)
        tk.Button(btn_frame, text="Analyze", command=self.run_analysis).pack(side="left", padx=6)
        tk.Button(btn_frame, text="Open Verse Manager", command=self.open_verse_manager).pack(side="left")
        tk.Button(btn_frame, text="Recommend Verses", command=self.recommend_from_input).pack(side="left", padx=6)

        # output panels
        tk.Label(root, text="Tokens (token -> type):").pack(anchor="w")
        self.token_box = scrolledtext.ScrolledText(root, height=6)
        self.token_box.pack(fill="both", padx=5, pady=4)

        tk.Label(root, text="Token Statistics:").pack(anchor="w")
        self.stat_box = scrolledtext.ScrolledText(root, height=4)
        self.stat_box.pack(fill="both", padx=5, pady=4)

        tk.Label(root, text="Sentiment:").pack(anchor="w")
        self.sentiment_box = scrolledtext.ScrolledText(root, height=4)
        self.sentiment_box.pack(fill="both", padx=5, pady=4)

        # verses dataset
        self.verses = []  # list of verse dicts
        self.verse_manager_window = None

    def run_analysis(self):
        text = self.input_box.get("1.0", tk.END).strip()
        strip = self.strip_var.get()
        ascii_map = self.ascii_var.get()
        tokens, types = tokenize(text, strip_diac=strip, ascii_map=ascii_map)
        stats = token_stats(tokens, types)

        # English sentiment based on english tokens
        eng_words = [t for t in tokens if ENGLISH_WORD_RE.fullmatch(t)]
        eng_text = " ".join(eng_words)
        label, scores = sentiment_label_and_scores_from_text_english(eng_text)

        # populate UI
        self.token_box.delete("1.0", tk.END)
        self.token_box.insert(tk.END, "\n".join(f"{tok}  -> {tt}" for tok, tt in zip(tokens, types)))

        self.stat_box.delete("1.0", tk.END)
        for k, v in stats.items():
            self.stat_box.insert(tk.END, f"{k}: {v}\n")

        self.sentiment_box.delete("1.0", tk.END)
        self.sentiment_box.insert(tk.END, f"{label}\nScores: {scores}")

    # ---------------------------
    # Verse Manager window & functionality
    # ---------------------------
    def open_verse_manager(self):
        if self.verse_manager_window and tk.Toplevel.winfo_exists(self.verse_manager_window):
            self.verse_manager_window.lift(); return

        w = tk.Toplevel(self.root)
        w.title("Verse Manager")
        self.verse_manager_window = w

        left = tk.Frame(w)
        left.pack(side="left", fill="y", padx=6, pady=6)

        right = tk.Frame(w)
        right.pack(side="left", fill="both", expand=True, padx=6, pady=6)

        # listbox of verses
        tk.Label(left, text="Verses:").pack(anchor="w")
        self.verse_listbox = tk.Listbox(left, width=40)
        self.verse_listbox.pack(fill="y", expand=True)
        self.verse_listbox.bind("<<ListboxSelect>>", self.on_verse_select)

        # verse operations
        op_frame = tk.Frame(left)
        op_frame.pack(fill="x", pady=4)
        tk.Button(op_frame, text="Load JSON", command=self.load_verses_from_file).pack(side="left", padx=2)
        tk.Button(op_frame, text="Save JSON", command=self.save_verses_to_file).pack(side="left", padx=2)
        tk.Button(op_frame, text="Add Verse", command=self.add_verse_dialog).pack(side="left", padx=2)
        tk.Button(op_frame, text="Delete", command=self.delete_selected_verse).pack(side="left", padx=2)

        tk.Button(left, text="Compute Sentiment for All", command=self.compute_sentiment_all).pack(fill="x", pady=6)

        # right: detail / preview / edit
        tk.Label(right, text="Verse Preview / Edit").pack(anchor="w")
        self.preview_box = scrolledtext.ScrolledText(right, height=12)
        self.preview_box.pack(fill="both", expand=True, padx=2, pady=2)

        detail_buttons = tk.Frame(right)
        detail_buttons.pack(fill="x", pady=4)
        tk.Button(detail_buttons, text="Save Edit", command=self.save_preview_edit).pack(side="left", padx=4)
        tk.Button(detail_buttons, text="Insert to Input", command=self.insert_selected_verse_to_input).pack(side="left", padx=4)
        tk.Button(detail_buttons, text="Recommend Similar", command=self.recommend_from_selected_verse).pack(side="left", padx=4)

        tk.Label(right, text="Recommendation Results:").pack(anchor="w", pady=(6,0))
        self.reco_box = scrolledtext.ScrolledText(right, height=8)
        self.reco_box.pack(fill="both", expand=True, padx=2, pady=2)

        # populate listbox initially
        self.refresh_verse_listbox()

    def refresh_verse_listbox(self):
        if not hasattr(self, 'verse_listbox') or self.verse_listbox is None:
            return
        self.verse_listbox.delete(0, tk.END)
        for i, v in enumerate(self.verses):
            title = v.get("title") or f"Verse {i+1}"
            vid = v.get("id") or v.get("verse_order") or i+1
            display = f"[{vid}] {title}"
            # show sentiment if computed
            lbl = v.get("sentiment_label")
            if lbl:
                display += f"  ({lbl})"
            self.verse_listbox.insert(tk.END, display)

    def load_verses_from_file(self):
        path = filedialog.askopenfilename(filetypes=[("JSON files","*.json"),("All files","*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and data.get('verses'):
                self.verses = data['verses']
            elif isinstance(data, list):
                self.verses = data
            else:
                messagebox.showerror("Format error", "JSON must be a list of verse objects or {verses: [...]}")
                return
            # compute sentiments lazily (or now)
            for v in self.verses:
                lbl, scores = compute_verse_sentiment(v)
                v['sentiment_label'] = lbl
                v['sentiment_scores'] = scores
            self.refresh_verse_listbox()
            messagebox.showinfo("Loaded", f"Loaded {len(self.verses)} verses.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load: {e}")

    def save_verses_to_file(self):
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files","*.json"),("All files","*.*")])
        if not path:
            return
        try:
            # don't overwrite sentiment fields if user prefers; we save full objects
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.verses, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("Saved", f"Saved {len(self.verses)} verses to {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")

    def add_verse_dialog(self):
        # simple dialog to add minimal verse fields. For more fields, user can edit the JSON in preview.
        new = {
            "chapter_id": simpledialog.askinteger("chapter_id", "chapter_id (integer)", initialvalue=1),
            "chapter_number": simpledialog.askinteger("chapter_number", "chapter_number", initialvalue=1),
            "externalId": simpledialog.askinteger("externalId", "externalId", initialvalue=len(self.verses)+1),
            "id": simpledialog.askinteger("id", "id", initialvalue=len(self.verses)+1),
            "text": simpledialog.askstring("text", "Devanagari text (full verse)"),
            "title": simpledialog.askstring("title", "Title (e.g. Verse 1)"),
            "verse_number": simpledialog.askinteger("verse_number", "verse_number", initialvalue=1),
            "verse_order": simpledialog.askinteger("verse_order", "verse_order", initialvalue=1),
            "transliteration": simpledialog.askstring("transliteration", "transliteration (IAST)"),
            "word_meanings": simpledialog.askstring("word_meanings", "word_meanings (English, plain text)")
        }
        # compute sentiment
        lbl, scores = compute_verse_sentiment(new)
        new['sentiment_label'] = lbl
        new['sentiment_scores'] = scores
        self.verses.append(new)
        self.refresh_verse_listbox()

    def on_verse_select(self, evt=None):
        sel = self.verse_listbox.curselection()
        if not sel: return
        idx = sel[0]
        v = self.verses[idx]
        # pretty print JSON-ish preview
        pretty = json.dumps(v, ensure_ascii=False, indent=2)
        self.preview_box.delete("1.0", tk.END)
        self.preview_box.insert(tk.END, pretty)

    def save_preview_edit(self):
        sel = self.verse_listbox.curselection()
        if not sel:
            messagebox.showwarning("Select", "Select a verse first.")
            return
        idx = sel[0]
        raw = self.preview_box.get("1.0", tk.END).strip()
        try:
            obj = json.loads(raw)
            # recompute sentiment
            lbl, scores = compute_verse_sentiment(obj)
            obj['sentiment_label'] = lbl
            obj['sentiment_scores'] = scores
            self.verses[idx] = obj
            self.refresh_verse_listbox()
            messagebox.showinfo("Saved", "Verse updated.")
        except Exception as e:
            messagebox.showerror("JSON error", f"Could not parse JSON: {e}")

    def delete_selected_verse(self):
        sel = self.verse_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        v = self.verses[idx]
        if messagebox.askyesno("Delete", f"Delete {v.get('title','verse')}?"):
            del self.verses[idx]
            self.refresh_verse_listbox()
            self.preview_box.delete("1.0", tk.END)

    def compute_sentiment_all(self):
        for v in self.verses:
            lbl, sc = compute_verse_sentiment(v)
            v['sentiment_label'] = lbl
            v['sentiment_scores'] = sc
        self.refresh_verse_listbox()
        messagebox.showinfo("Done", "Recomputed sentiment for all verses.")

    def insert_selected_verse_to_input(self):
        sel = self.verse_listbox.curselection()
        if not sel:
            messagebox.showwarning("Select", "Select a verse first.")
            return
        idx = sel[0]
        v = self.verses[idx]
        # insert devanagari 'text' if present else transliteration
        text = v.get("text") or v.get("transliteration") or ""
        self.input_box.delete("1.0", tk.END)
        self.input_box.insert(tk.END, text)
        messagebox.showinfo("Inserted", "Verse inserted into input box.")

    def recommend_from_selected_verse(self):
        sel = self.verse_listbox.curselection()
        if not sel:
            messagebox.showwarning("Select", "Select a verse first.")
            return
        idx = sel[0]
        v = self.verses[idx]
        # build a short text combining transliteration/word_meanings for recommendation seed
        seed = v.get("word_meanings") or v.get("transliteration") or v.get("text") or ""
        recs = recommend_verses_for_input(seed, self.verses,
                                          strip_diac=self.strip_var.get(),
                                          ascii_map=self.ascii_var.get(), top_n=6)
        self.reco_box.delete("1.0", tk.END)
        for r_v, score in recs:
            self.reco_box.insert(tk.END, f"Score {score:.3f} | [{r_v.get('id')}] {r_v.get('title')} ({r_v.get('sentiment_label')})\n")
            # one-line preview
            one_line = (r_v.get("text") or r_v.get("transliteration") or "").splitlines()[0][:200]
            self.reco_box.insert(tk.END, f"    {one_line}\n\n")

    # ---------------------------
    # Recommendation from the main input text
    # ---------------------------
    def recommend_from_input(self):
        text = self.input_box.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Input empty", "Write something in the input box to get recommendations.")
            return
        if not self.verses:
            messagebox.showwarning("No verses", "Load or add verses in the Verse Manager first.")
            return
        recs = recommend_verses_for_input(text, self.verses,
                                          strip_diac=self.strip_var.get(),
                                          ascii_map=self.ascii_var.get(), top_n=6)
        # show result in a dialog
        out = []
        for v, score in recs:
            out.append({
                "id": v.get("id"),
                "title": v.get("title"),
                "sentiment": v.get("sentiment_label"),
                "score": score,
                "text_preview": (v.get("text") or v.get("transliteration") or "").splitlines()[0][:200]
            })
        # show in a simple results window
        res_w = tk.Toplevel(self.root); res_w.title("Recommendations")
        tk.Label(res_w, text=f"Top {len(out)} matches:").pack(anchor="w")
        box = scrolledtext.ScrolledText(res_w, height=15, width=80)
        box.pack(fill="both", padx=6, pady=6)
        for item in out:
            box.insert(tk.END, f"[{item['id']}] {item['title']}  ({item['sentiment']}, score={item['score']:.3f})\n")
            box.insert(tk.END, f"    {item['text_preview']}\n\n")
        # optionally allow user to insert any recommended verse to input by clicking on the list -> keep simple: ask id
        def insert_by_id():
            v_id = simpledialog.askstring("Insert by id", "Enter id of verse to insert into input")
            if not v_id: return
            for vv in self.verses:
                if str(vv.get('id')) == str(v_id) or str(vv.get('verse_order')) == str(v_id):
                    self.input_box.delete("1.0", tk.END)
                    self.input_box.insert(tk.END, vv.get("text") or vv.get("transliteration") or "")
                    messagebox.showinfo("Inserted", f"Inserted {vv.get('title')}")
                    res_w.destroy()
                    return
            messagebox.showerror("Not found", "Could not find verse with that id.")
        tk.Button(res_w, text="Insert a recommended verse by ID", command=insert_by_id).pack(pady=4)

# ---------------------------
# Run application
# ---------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = GitaApp(root)
    root.mainloop()
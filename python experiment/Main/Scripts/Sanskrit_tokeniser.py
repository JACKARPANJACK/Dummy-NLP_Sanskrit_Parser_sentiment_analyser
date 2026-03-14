# main.py
"""
Bhagavad Gita — Interactive Wisdom Engine (complete)
Features:
 - Devanagari + Latin tokenizer
 - Token counts & types
 - Combined sentiment (VADER + tiny Hindi lexicon boost)
 - TF-IDF and optional semantic search (sentence-transformers)
 - Verse Manager (load/save/edit JSON)
 - Interactive recommendation table
 - Scripture-style Reader (Sanskrit left, Translation+Commentary right)
 - Prev/Next/Jump, font slider, copy translation, save explanation
 - TTS via pyttsx3 (optional)
 - Krishna Chat: grounded replies using recommended verses
 - Bookmarks + save bookmarks
"""

import os
import sys
import re
import json
import unicodedata
import threading
import pickle
import random
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox, simpledialog

# NLP
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional: TTS
try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    pyttsx3 = None
    TTS_AVAILABLE = False

# Optional: Semantic model
SEMANTIC_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_AVAILABLE = True
except Exception:
    SEMANTIC_AVAILABLE = False

# -------------------------------------------------------------------------
# Resource helper (for packaging with PyInstaller)
# -------------------------------------------------------------------------
def resource_path(relative):
    try:
        base_path = sys._MEIPASS  # pyinstaller
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative)

# -------------------------------------------------------------------------
# NLTK/VADER setup
# -------------------------------------------------------------------------
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()

# -------------------------------------------------------------------------
# UI theme (Veda-inspired)
# -------------------------------------------------------------------------
BG_COLOR = "#f5e6c8"
FRAME_COLOR = "#c48a2c"
TEXT_COLOR = "#3b2b1a"
GOLD = "#d4af37"
CARD_BG = "#fffaf0"

# -------------------------------------------------------------------------
# Unicode helpers & Devanagari mapping (small transliterator)
# -------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", (text or ""))

DEV_RANGE = r"\u0900-\u097F"
DEVANAGARI_RE = re.compile(rf"[{DEV_RANGE}]")

# Word regex: Latin words with hyphen/apostrophe OR contiguous Devanagari cluster
WORD_RE = re.compile(rf"[A-Za-z]+(?:[-'][A-Za-z]+)*|[{DEV_RANGE}]+")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
# include danda (।) explicitly, plus common punctuation
PUNCT_RE = re.compile(r"[।\.\?\!\,\;\:\—\-\"\'\(\)\[\]<>]|[^\s\w]")

SENT_SPLIT_RE = re.compile(r"[।\?\!\.]+")  # sentence split on danda or punctuation

# Small Devanagari→Latin mapping for romanization (approx)
DEV_TO_LATIN = {
    "अ":"a","आ":"ā","इ":"i","ई":"ī","उ":"u","ऊ":"ū","ऋ":"ṛ","ॠ":"ṝ",
    "ए":"e","ऐ":"ai","ओ":"o","औ":"au",
    "ं":"ṃ","ः":"ḥ","ँ":"̃",
    "क":"k","ख":"kh","ग":"g","घ":"gh","ङ":"ṅ",
    "च":"c","छ":"ch","ज":"j","झ":"jh","ञ":"ñ",
    "ट":"ṭ","ठ":"ṭh","ड":"ḍ","ढ":"ḍh","ण":"ṇ",
    "त":"t","थ":"th","द":"d","ध":"dh","न":"n",
    "प":"p","फ":"ph","ब":"b","भ":"bh","म":"m",
    "य":"y","र":"r","ल":"l","व":"v",
    "श":"ś","ष":"ṣ","स":"s","ह":"h",
    "ळ":"ḷ","क्ष":"kṣ","त्र":"tr","ज्ञ":"jñ",
    "ा":"ā","ि":"i","ी":"ī","ु":"u","ू":"ū","ृ":"ṛ","ॄ":"ṝ","े":"e","ै":"ai","ो":"o","ौ":"au",
    "्":"", "़":"", "।":"."
}

def devanagari_to_latin(text: str) -> str:
    text = normalize_text(text or "")
    out = []
    i = 0
    L = len(text)
    while i < L:
        # try two-character conjuncts first
        if i+1 < L and text[i:i+2] in DEV_TO_LATIN:
            out.append(DEV_TO_LATIN[text[i:i+2]])
            i += 2
            continue
        ch = text[i]
        if ch in DEV_TO_LATIN:
            out.append(DEV_TO_LATIN[ch])
        else:
            # keep ascii as-is
            if ord(ch) < 128:
                out.append(ch)
            # else skip unknown combining signs
        i += 1
    s = "".join(out)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def contains_devanagari(text: str) -> bool:
    return bool(DEVANAGARI_RE.search(text or ""))

def split_sentences(text: str):
    if not text:
        return []
    parts = SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]

# -------------------------------------------------------------------------
# Tokenizer (returns tokens list and types list)
# -------------------------------------------------------------------------
def tokenize(text: str):
    text = normalize_text(text or "")
    tokens = []
    types = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        m = NUMBER_RE.match(text, i)
        if m:
            tokens.append(m.group(0)); types.append("NUMBER"); i = m.end(); continue
        m = WORD_RE.match(text, i)
        if m:
            tok = m.group(0)
            if contains_devanagari(tok):
                types.append("DEVANAGARI_WORD")
            else:
                types.append("WORD")
            tokens.append(tok); i = m.end(); continue
        m = PUNCT_RE.match(text, i)
        if m:
            tokens.append(m.group(0)); types.append("PUNCT"); i = m.end(); continue
        tokens.append(ch); types.append("UNKNOWN"); i += 1
    return tokens, types

# token stats
from collections import Counter
def get_token_stats(tokens, types):
    c = Counter(types)
    return {
        "TOTAL": len(tokens),
        "WORD": c.get("WORD", 0),
        "DEVANAGARI": c.get("DEVANAGARI_WORD", 0),
        "NUMBER": c.get("NUMBER", 0),
        "PUNCT": c.get("PUNCT", 0),
        "UNKNOWN": c.get("UNKNOWN", 0)
    }

# -------------------------------------------------------------------------
# Tiny Hindi sentiment lexicon to nudge VADER for Devanagari inputs
# -------------------------------------------------------------------------
HINDI_POS = {"खुश","सुख","आनंद","प्रेम","प्यार","शुभ","संतोष","धन्य","जीत","विजय","साहस"}
HINDI_NEG = {"डर","भय","दुःख","दुख","रोग","शोक","पराजय","हार","क्रोध","उदास","डरना"}

def hindi_lexicon_score(text: str) -> float:
    if not text:
        return 0.0
    s = 0.0
    for w in HINDI_POS:
        if w in text:
            s += 0.25
    for w in HINDI_NEG:
        if w in text:
            s -= 0.25
    return s

# -------------------------------------------------------------------------
# Combined sentiment function: supports Devanagari
# returns (label, compound_score, details)
# -------------------------------------------------------------------------
def combined_sentiment(text: str):
    if not text:
        return "Neutral", 0.0, {}
    text = str(text)
    if contains_devanagari(text):
        sents = split_sentences(text)
        compounds = []
        details = {"by_sentence": []}
        for sent in sents:
            latin = devanagari_to_latin(sent)
            vader_scores = sia.polarity_scores(latin)
            base = vader_scores.get("compound", 0.0)
            boost = hindi_lexicon_score(sent)
            compound = max(-1.0, min(1.0, base + boost))
            compounds.append(compound)
            details["by_sentence"].append({
                "sentence": sent,
                "latin": latin,
                "vader": vader_scores,
                "boost": boost,
                "compound": compound
            })
        avg = sum(compounds)/len(compounds) if compounds else 0.0
        if avg >= 0.05:
            label = "Positive"
        elif avg <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"
        return label, avg, details
    else:
        words = re.findall(r"[A-Za-z']+", text)
        if not words:
            return "Neutral", 0.0, {}
        txt = " ".join(words)
        scores = sia.polarity_scores(txt)
        c = scores.get("compound", 0.0)
        if c >= 0.05:
            label = "Positive"
        elif c <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"
        return label, c, {"vader": scores}

# -------------------------------------------------------------------------
# Mood detection (English + Hindi keyword mapping)
# -------------------------------------------------------------------------
MOOD_KEYWORDS = {
    "fear":["fear","afraid","panic","anxious","डर","भय","डरना"],
    "courage":["fight","battle","brave","strength","साहस","विजय"],
    "dharma":["duty","responsibility","justice","कर्तव्य","धर्म"],
    "detachment":["let go","loss","attachment","desire","असक्ति","छूट"],
    "devotion":["love","faith","god","krishna","प्रेम","भक्ति"],
    "wisdom":["knowledge","truth","learn","ज्ञान","सत्य"],
    "peace":["calm","peace","serenity","meditation","शान्ति","शांति"]
}
def detect_mood(text: str) -> str:
    if not text:
        return "wisdom"
    t = text.lower()
    for mood, words in MOOD_KEYWORDS.items():
        for w in words:
            if w in t:
                return mood
    return "wisdom"

# -------------------------------------------------------------------------
# Verses storage and indexing (TF-IDF + optional Semantic)
# -------------------------------------------------------------------------
verses = []              # list of verse dicts
tf_vectorizer = None
tf_matrix = None

semantic_model = None
semantic_embeddings = None

def build_tfidf_index(verses_list):

    global tf_vectorizer, tf_matrix

    corpus = []

    for v in verses_list:

        text_parts = []

        if v.get("translation"):
            text_parts.append(v["translation"])

        if v.get("word_meanings"):
            text_parts.append(v["word_meanings"])

        if v.get("transliteration"):
            text_parts.append(v["transliteration"])
            
        if v.get("commentary_shankara"):
            text_parts.append(v["commentary_shankara"])
            
        if v.get("commentary_ramanuja"):
            text_parts.append(v["commentary_ramanuja"])
            
        if v.get("commentary_madhva"):
            text_parts.append(v["commentary_madhva"])

        corpus.append(" ".join(text_parts))

    if not corpus:
        tf_vectorizer = None
        tf_matrix = None
        return

    tf_vectorizer = TfidfVectorizer(stop_words="english")

    tf_matrix = tf_vectorizer.fit_transform(corpus)

    print("TF-IDF index built for", len(corpus), "verses")

def recommend_tfidf(query, top_n=6):

    global tf_vectorizer, tf_matrix

    if tf_vectorizer is None or tf_matrix is None:
        return []

    q = query

    if contains_devanagari(query):
        q = devanagari_to_latin(query) + " " + query

    q_vec = tf_vectorizer.transform([q.lower()])

    similarities = cosine_similarity(q_vec, tf_matrix)[0]

    idxs = similarities.argsort()[::-1][:top_n]

    results = []

    for i in idxs:
        results.append((verses[i], float(similarities[i])))

    return results

# semantic index
def try_build_semantic_index():
    global semantic_model, semantic_embeddings
    if not SEMANTIC_AVAILABLE:
        semantic_model = None
        semantic_embeddings = None
        return False
    try:
        semantic_model = SentenceTransformer("all-MiniLM-L6-v2")  # small & quick
        corpus = []
        for v in verses:
            text = ""
            if v.get("translation"):
                text += v["translation"] + " "
            if v.get("word_meanings"):
                text += v["word_meanings"] + " "
            if v.get("transliteration"):
                text += v.get("transliteration")
            corpus.append(text.strip())
        if corpus:
            semantic_embeddings = semantic_model.encode(corpus, show_progress_bar=False, convert_to_numpy=True)
        else:
            semantic_embeddings = None
        return True
    except Exception as e:
        semantic_model = None
        semantic_embeddings = None
        return False

def recommend_semantic(query: str, top_n=6):
    global semantic_model, semantic_embeddings
    if not semantic_model or semantic_embeddings is None:
        return []
    q = query
    if contains_devanagari(query):
        q = devanagari_to_latin(query) + " " + re.sub(r"[^\w\s']", " ", query)
    q_emb = semantic_model.encode([q], convert_to_numpy=True)
    sim = cosine_similarity(q_emb, semantic_embeddings)[0]
    idxs = sim.argsort()[::-1][:top_n]
    return [(verses[i], float(sim[i])) for i in idxs]

# -------------------------------------------------------------------------
# JSON load/save utilities (packaging-safe)
# -------------------------------------------------------------------------
def load_verses_from_path(path):
    global verses, tf_vectorizer, tf_matrix
    try:
        base_dir = os.path.dirname(resource_path(path) if getattr(sys, "_MEIPASS", None) else path)
        real_path = resource_path(path) if getattr(sys, "_MEIPASS", None) else path

        cache_path = os.path.join(base_dir, "verses_cache.pkl")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    cache_data = pickle.load(f)
                verses = cache_data.get("verses", [])
                tf_vectorizer = cache_data.get("tf_vectorizer")
                tf_matrix = cache_data.get("tf_matrix")
                print("Loaded verses and TF-IDF from cache successfully!")
                threading.Thread(target=try_build_semantic_index, daemon=True).start()
                return True, len(verses)
            except Exception as e:
                print(f"Notice: Cache load failed: {e}")

        with open(real_path, "r", encoding="utf8") as f:
            data = json.load(f)
            if isinstance(data, dict) and data.get("verses"):
                verses = data["verses"]
            elif isinstance(data, list):
                verses = data
            else:
                raise ValueError("JSON must be list or {verses:[...]} format")
        
        # Merge translations from translation.json if available
        trans_path = os.path.join(base_dir, "translation.json")
        if not os.path.exists(trans_path):
            trans_path = resource_path("translation.json") if getattr(sys, "_MEIPASS", None) else "translation.json"
            
        if os.path.exists(trans_path):
            try:
                with open(trans_path, "r", encoding="utf8") as tf:
                    t_data = json.load(tf)
                # Map translations by verse_id, preferring English
                trans_map = {}
                for t in t_data:
                    # Filter for english translations
                    if t.get("lang", "").lower() == "english":
                        vid = t.get("verse_id")
                        if vid not in trans_map:
                            trans_map[vid] = []
                        trans_map[vid].append(t.get("description", "").strip())
                
                # Apply translations to verses
                for v in verses:
                    vid = v.get("id")
                    if vid in trans_map:
                        v["translation"] = "\n\n".join(trans_map[vid])
            except Exception as e:
                print(f"Warning: could not load translations from {trans_path}: {e}")

        # Merge commentaries from commentary.json if available
        comm_path = os.path.join(base_dir, "commentary.json")
        if not os.path.exists(comm_path):
            comm_path = resource_path("commentary.json") if getattr(sys, "_MEIPASS", None) else "commentary.json"

        if os.path.exists(comm_path):
            try:
                with open(comm_path, "r", encoding="utf8") as cf:
                    c_data = json.load(cf)
                
                comm_map = {}
                for c in c_data:
                    vid = c.get("verse_id")
                    author = c.get("authorName", "")
                    lang = c.get("lang", "").lower()
                    
                    if vid not in comm_map:
                        comm_map[vid] = {"shankara": [], "ramanuja": [], "madhva": []}
                        
                    desc = c.get("description", "").strip()
                    if desc:
                        # Append with language tag for better readability
                        text_entry = f"[{lang.upper()}]\n{desc}"
                        if "Shankaracharya" in author:
                            comm_map[vid]["shankara"].append(text_entry)
                        elif "Ramanujacharya" in author:
                            comm_map[vid]["ramanuja"].append(text_entry)
                        elif "Madhavacharya" in author:
                            comm_map[vid]["madhva"].append(text_entry)

                # Apply commentaries to verses
                for v in verses:
                    vid = v.get("id")
                    if vid in comm_map:
                        c_m = comm_map[vid]
                        if c_m["shankara"]:
                            v["commentary_shankara"] = "\n\n".join(c_m["shankara"])
                        if c_m["ramanuja"]:
                            v["commentary_ramanuja"] = "\n\n".join(c_m["ramanuja"])
                        if c_m["madhva"]:
                            v["commentary_madhva"] = "\n\n".join(c_m["madhva"])

            except Exception as e:
                print(f"Warning: could not load commentaries from {comm_path}: {e}")

        build_tfidf_index(verses)
        
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({
                    "verses": verses,
                    "tf_vectorizer": tf_vectorizer,
                    "tf_matrix": tf_matrix
                }, f)
            print("Successfully cached verses and internal matrix.")
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

        # attempt semantic index in background
        threading.Thread(target=try_build_semantic_index, daemon=True).start()
        return True, len(verses)
    except Exception as e:
        return False, str(e)


def save_verses_to_path(path):
    global verses
    try:
        with open(path, "w", encoding="utf8") as f:
            json.dump(verses, f, ensure_ascii=False, indent=2)
        return True, None
    except Exception as e:
        return False, str(e)

# -------------------------------------------------------------------------
# TTS helper
# -------------------------------------------------------------------------
tts_engine = None
if TTS_AVAILABLE:
    try:
        tts_engine = pyttsx3.init()
    except Exception:
        tts_engine = None
        TTS_AVAILABLE = False

def speak_text_async(text: str):
    if not TTS_AVAILABLE or not tts_engine or not text:
        return
    def _run():
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True).start()

# -------------------------------------------------------------------------
# Krishna Chat: simple grounded "sage" replies using best-matching verses
# (no LLM calls — uses templates + verse grounding)
# -------------------------------------------------------------------------
def krishna_reply(user_text: str, top_k=3):
    # detect mood & sentiment
    label, compound, details = combined_sentiment(user_text)
    mood = detect_mood(user_text)
    
    # Try semantic first
    recs = recommend_semantic(user_text, top_n=top_k) if semantic_model and semantic_embeddings is not None else []
    if not recs:
        recs = recommend_tfidf(user_text, top_n=top_k)
        
    # Build a textual response
    parts = []
    
    intros = [
        "कृष्ण: मित्र / O friend, reflect and listen. The truth is eternal. (友よ、反省して聞いてください。)",
        "Krishna: Mein lieber Freund (My dear friend), do not let your mind be clouded by illusions. Here is what you must understand:",
        "कृष्ण: 友よ (Tomo yo - Friend), the material world is fleeting. Listen to the wisdom of the ages:",
        "Krishna: Seien Sie standhaft! (Be steadfast!) O Arjuna, steady your mind and hear these words:",
        "कृष्ण: पार्थ (O Partha), in times of doubt, true knowledge is the only refuge. 聞いてください (Listen):",
        "Krishna: Der Geist ist unruhig, aber das Wissen bringt Frieden. (The mind is restless, but wisdom brings peace). O friend, observe:"
    ]
    
    import random
    parts.append(random.choice(intros) + "\n")
    
    for (v,score) in recs:
        ch = v.get("chapter_number", "?")
        vn = v.get("verse_number", "?")
        title = v.get("title") or f"{ch}:{vn}"
        trans = v.get("translation") or v.get("word_meanings") or v.get("transliteration") or ""
        short = (trans.splitlines()[0] if trans else "")[:250]
        parts.append(f"✦ Bhagavad Gita {ch}.{vn} ({title}) \n   {short} [relevance={score:.3f}]")

    # Elaborate interpretation matching Lord Krishna's style with mixed languages
    if recs:
        parts.append("\n[कृष्ण का मार्गदर्शन / Lord's Guidance]")
        if mood == "fear":
            comment = (
                "Why this fear, O friend? Furcht ist nur eine Illusion (Fear is but an illusion). "
                "The soul is never born, nor does it die. "
                "恐れないでください (Do not fear). Perform your duty without attachment to the outcome, "
                "for the eternal self is indestructible."
            )
        elif mood == "dharma":
            comment = (
                "Your focus on duty is righteous, but do so selflessly. "
                "Tue deine Pflicht ohne Anhaftung (Do your duty without attachment). "
                "結果に執着せず、義務を果たす (Fulfill your duty without clinging to the results). "
                "True Dharma is acting for the Supreme, offering all actions to me."
            )
        elif mood == "devotion":
            comment = (
                "Your heart shows the path of Bhakti (Devotion). "
                "Hingabe an das Göttliche befreit die Seele (Devotion to the Divine frees the soul). "
                "私への信愛 (Faith in me) is the highest path. Those who surrender unto Me, I swiftly deliver from the ocean of birth and death."
            )
        elif mood == "detachment":
            comment = (
                "Vairagya (Detachment) is the sword that cuts the binding ropes of Maya. "
                "Lass los und finde Frieden (Let go and find peace). "
                "執着を手放す (Let go of attachment). When the mind is untouched by sensory pleasures, one reaches true equanimity."
            )
        else:
            comment = (
                "Read these verses and meditate upon them. "
                "Wahre Weisheit kommt von innen (True wisdom comes from within). "
                "心の中で瞑想しなさい (Meditate within your heart). "
                "He who sees Me in everything, and everything in Me, is never separated from Me."
            )
        parts.append(" " + comment + "\n")
    else:
        parts.append("I couldn't find a matching verse. 落ち着いて (Stay calm), and seek your answer within, oder frag mich auf andere Weise (or ask me differently).")
        
    return "\n".join(parts)

# -------------------------------------------------------------------------   
# GUI application classes
# -------------------------------------------------------------------------   
class ReaderWindow:
    """Scripture-style reader: left Sanskrit, right translation + commentary"""
    def __init__(self, parent, verse_index, app_ref):
        self.app = app_ref
        self.index = verse_index
        self.verse = verses[self.index]
        self.win = tk.Toplevel(parent)
        self.win.title(f"Gita Reader — {self.verse.get('chapter_number','?')}:{self.verse.get('verse_number','?')}")
        self.win.configure(bg=BG_COLOR)
        self.win.geometry("1000x700")

        # Header bar
        header = tk.Frame(self.win, bg=BG_COLOR)
        header.pack(fill="x", padx=8, pady=6)
        title = tk.Label(header, text=f"Chapter {self.verse.get('chapter_number','?')}  Verse {self.verse.get('verse_number','?')}",
                         font=("Noto Serif Devanagari", 16, "bold"), bg=BG_COLOR, fg=GOLD)
        title.pack(side="left", padx=(4,10))

        btn_prev = tk.Button(header, text="◀ Prev", bg=FRAME_COLOR, fg="white", command=self.prev_verse)
        btn_prev.pack(side="left", padx=4)
        btn_next = tk.Button(header, text="Next ▶", bg=FRAME_COLOR, fg="white", command=self.next_verse)
        btn_next.pack(side="left", padx=4)

        btn_chapter = tk.Button(header, text="Jump to...", bg=FRAME_COLOR, fg="white", command=self.jump_to)
        btn_chapter.pack(side="left", padx=4)

        tk.Button(header, text="Copy Translation", bg=FRAME_COLOR, fg="white", command=self.copy_translation).pack(side="left", padx=6)
        tk.Button(header, text="Save Edits", bg=FRAME_COLOR, fg="white", command=self.save_edits).pack(side="left", padx=6)
        if TTS_AVAILABLE:
            tk.Button(header, text="🔊 Speak Translation", bg=FRAME_COLOR, fg="white", command=self.speak_translation).pack(side="left", padx=6)
            tk.Button(header, text="🔊 Chant (translit)", bg=FRAME_COLOR, fg="white", command=self.speak_transliteration).pack(side="left", padx=6)

        # Font slider
        self.font_size = tk.IntVar(value=16)
        tk.Label(header, text="Font:", bg=BG_COLOR).pack(side="right", padx=(2,0))
        tk.Scale(header, from_=12, to=26, orient="horizontal", variable=self.font_size, length=140, bg=BG_COLOR, command=lambda e: self.render()).pack(side="right", padx=8)

        # Main content area: left/right
        main = tk.Frame(self.win, bg=BG_COLOR)
        main.pack(fill="both", expand=True, padx=8, pady=6)

        left = tk.Frame(main, bg=CARD_BG)
        left.pack(side="left", fill="both", expand=True, padx=4, pady=4)

        right = tk.Frame(main, bg=CARD_BG)
        right.pack(side="left", fill="both", expand=True, padx=4, pady=4)

        # Sanskrit panel
        tk.Label(left, text="🕉 Sanskrit (Devanagari)", bg=CARD_BG, font=("Noto Serif Devanagari", 12, "bold")).pack(anchor="w", padx=6, pady=(6,0))
        self.sanskrit_area = scrolledtext.ScrolledText(left, wrap=tk.WORD, font=("Noto Serif Devanagari", self.font_size.get()), bg=CARD_BG)
        self.sanskrit_area.pack(fill="both", expand=True, padx=6, pady=6)

        # Translation + tabs (translation and commentaries)
        tk.Label(right, text="🌍 Translation / Explanation", bg=CARD_BG, font=("Georgia", 12, "bold")).pack(anchor="w", padx=6, pady=(6,0))
        self.translation_area = scrolledtext.ScrolledText(right, wrap=tk.WORD, font=("Georgia", 14), bg=CARD_BG, height=10)
        self.translation_area.pack(fill="both", expand=False, padx=6, pady=6)

        # Notebook for commentaries
        self.notebook = ttk.Notebook(right)
        self.tab_shankara = tk.Frame(self.notebook, bg=CARD_BG)
        self.tab_ramanuja = tk.Frame(self.notebook, bg=CARD_BG)
        self.tab_madhva = tk.Frame(self.notebook, bg=CARD_BG)
        self.tab_notes = tk.Frame(self.notebook, bg=CARD_BG)

        self.notebook.add(self.tab_shankara, text="Śaṅkara")
        self.notebook.add(self.tab_ramanuja, text="Rāmānuja")
        self.notebook.add(self.tab_madhva, text="Mādhva")
        self.notebook.add(self.tab_notes, text="Notes / Explanation")
        self.notebook.pack(fill="both", expand=True, padx=6, pady=(0,6))

        # Commentary text areas (editable for notes)
        self.c_shankara = scrolledtext.ScrolledText(self.tab_shankara, wrap=tk.WORD, bg=CARD_BG)
        self.c_shankara.pack(fill="both", expand=True, padx=6, pady=6)
        self.c_ramanuja = scrolledtext.ScrolledText(self.tab_ramanuja, wrap=tk.WORD, bg=CARD_BG)
        self.c_ramanuja.pack(fill="both", expand=True, padx=6, pady=6)
        self.c_madhva = scrolledtext.ScrolledText(self.tab_madhva, wrap=tk.WORD, bg=CARD_BG)
        self.c_madhva.pack(fill="both", expand=True, padx=6, pady=6)
        self.c_notes = scrolledtext.ScrolledText(self.tab_notes, wrap=tk.WORD, bg=CARD_BG)
        self.c_notes.pack(fill="both", expand=True, padx=6, pady=6)

        # Initialize content
        self.load_current_verse_content()
        self.render()

    def load_current_verse_content(self):
        v = self.verse
        # fill sanskrit area
        self.sanskrit_area.config(state="normal"); self.sanskrit_area.delete("1.0", tk.END)
        self.sanskrit_area.insert(tk.END, v.get("text","(No Sanskrit text provided)"))
        self.sanskrit_area.config(state="disabled")
        # translation
        self.translation_area.config(state="normal"); self.translation_area.delete("1.0", tk.END)
        self.translation_area.insert(tk.END, v.get("translation") or v.get("word_meanings", "(No translation provided)"))
        self.translation_area.config(state="disabled")
        # commentaries: load if present in verse dict (expected keys: commentary_shankara, commentary_ramanuja, commentary_madhva, explanation)
        self.c_shankara.delete("1.0", tk.END); self.c_shankara.insert(tk.END, v.get("commentary_shankara",""))
        self.c_ramanuja.delete("1.0", tk.END); self.c_ramanuja.insert(tk.END, v.get("commentary_ramanuja",""))
        self.c_madhva.delete("1.0", tk.END); self.c_madhva.insert(tk.END, v.get("commentary_madhva",""))
        self.c_notes.delete("1.0", tk.END); self.c_notes.insert(tk.END, v.get("explanation",""))

    def render(self):
        # adjust fonts
        fs = max(12, int(self.font_size.get()))
        self.sanskrit_area.configure(font=("Noto Serif Devanagari", fs))
        self.translation_area.configure(font=("Georgia", max(12, fs-2)))
        # ensure areas are readonly where appropriate
        # (translation area kept readonly to avoid accidental edits; notes are editable)
        self.translation_area.config(state="disabled")

    def prev_verse(self):
        if self.index > 0:
            self.index -= 1
            self.verse = verses[self.index]
            self.load_current_verse_content()
            self.render()

    def next_verse(self):
        if self.index < len(verses)-1:
            self.index += 1
            self.verse = verses[self.index]
            self.load_current_verse_content()
            self.render()

    def jump_to(self):
        # ask for chapter:verse like "2:47"
        s = simpledialog.askstring("Jump", "Enter Chapter:Verse (e.g. 2:47)")
        if not s: return
        m = re.match(r"\s*(\d+)\s*[:\-]\s*(\d+)\s*$", s)
        if not m:
            messagebox.showerror("Format", "Please use format CHAPTER:VERSE (e.g. 2:47)")
            return
        ch, vn = int(m.group(1)), int(m.group(2))
        for i, v in enumerate(verses):
            if int(v.get("chapter_number", -1)) == ch and int(v.get("verse_number", -1)) == vn:
                self.index = i
                self.verse = verses[self.index]
                self.load_current_verse_content()
                self.render()
                return
        messagebox.showinfo("Not found", f"Verse {ch}:{vn} not in loaded dataset.")

    def copy_translation(self):
        tr = self.verse.get("translation","")
        self.win.clipboard_clear()
        self.win.clipboard_append(tr)
        messagebox.showinfo("Copied", "Translation copied to clipboard.")

    def save_edits(self):
        # save commentaries and notes back to verses list and optionally persist later
        self.verse["commentary_shankara"] = self.c_shankara.get("1.0", tk.END).strip()
        self.verse["commentary_ramanuja"] = self.c_ramanuja.get("1.0", tk.END).strip()
        self.verse["commentary_madhva"] = self.c_madhva.get("1.0", tk.END).strip()
        self.verse["explanation"] = self.c_notes.get("1.0", tk.END).strip()
        # user feedback
        messagebox.showinfo("Saved", "Commentary and notes saved into memory. Use 'Save JSON' in main window to persist.")

    def speak_translation(self):
        txt = self.verse.get("translation") or ""
        if not txt:
            messagebox.showinfo("Nothing", "No translation available to speak.")
            return
        speak_text_async(txt)

    def speak_transliteration(self):
        txt = self.verse.get("transliteration") or self.verse.get("text") or ""
        if not txt:
            messagebox.showinfo("Nothing", "No transliteration/text available to chant.")
            return
        speak_text_async(txt)

class VerseManagerWindow:
    """A simple verse manager to add/edit/delete verses and save/load JSON"""
    def __init__(self, parent, app_ref):
        self.app = app_ref
        self.win = tk.Toplevel(parent)
        self.win.title("Verse Manager")
        self.win.geometry("900x600")
        self.win.configure(bg=BG_COLOR)
        # left list / right editor
        left = tk.Frame(self.win, bg=BG_COLOR)
        left.pack(side="left", fill="y", padx=6, pady=6)
        right = tk.Frame(self.win, bg=BG_COLOR)
        right.pack(side="left", fill="both", expand=True, padx=6, pady=6)

        tk.Label(left, text="Verses", bg=BG_COLOR, fg=TEXT_COLOR).pack(anchor="w")
        self.listbox = tk.Listbox(left, width=30, height=30)
        self.listbox.pack(fill="y", expand=False)
        self.listbox.bind("<<ListboxSelect>>", self.on_select)

        btns = tk.Frame(left, bg=BG_COLOR)
        btns.pack(fill="x", pady=6)
        tk.Button(btns, text="Add", bg=FRAME_COLOR, fg="white", command=self.add_verse).pack(side="left", padx=2)
        tk.Button(btns, text="Delete", bg=FRAME_COLOR, fg="white", command=self.delete_verse).pack(side="left", padx=2)
        tk.Button(btns, text="Save JSON", bg=FRAME_COLOR, fg="white", command=self.save_json).pack(side="left", padx=2)
        tk.Button(btns, text="Close", command=self.win.destroy).pack(side="left", padx=2)

        # editor fields on right
        form = tk.Frame(right, bg=BG_COLOR)
        form.pack(fill="both", expand=True)
        # columns: text areas for Devanagari, transliteration, translation, word_meanings
        tk.Label(form, text="Sanskrit (text)", bg=BG_COLOR).grid(row=0, column=0, sticky="w")
        self.txt_sanskrit = scrolledtext.ScrolledText(form, width=60, height=6, bg=CARD_BG)
        self.txt_sanskrit.grid(row=1, column=0, padx=4, pady=4, columnspan=2)

        tk.Label(form, text="Transliteration", bg=BG_COLOR).grid(row=2, column=0, sticky="w")
        self.txt_translit = scrolledtext.ScrolledText(form, width=60, height=3, bg=CARD_BG)
        self.txt_translit.grid(row=3, column=0, padx=4, pady=4, columnspan=2)

        tk.Label(form, text="Translation", bg=BG_COLOR).grid(row=4, column=0, sticky="w")
        self.txt_translation = scrolledtext.ScrolledText(form, width=60, height=6, bg=CARD_BG)
        self.txt_translation.grid(row=5, column=0, padx=4, pady=4, columnspan=2)

        tk.Label(form, text="Word Meanings (english)", bg=BG_COLOR).grid(row=6, column=0, sticky="w")
        self.txt_meanings = scrolledtext.ScrolledText(form, width=60, height=4, bg=CARD_BG)
        self.txt_meanings.grid(row=7, column=0, padx=4, pady=4, columnspan=2)

        # id / chapter / verse
        tk.Label(form, text="Chapter #", bg=BG_COLOR).grid(row=8, column=0, sticky="w")
        self.entry_chapter = tk.Entry(form)
        self.entry_chapter.grid(row=8, column=1, sticky="w")
        tk.Label(form, text="Verse #", bg=BG_COLOR).grid(row=9, column=0, sticky="w")
        self.entry_verse = tk.Entry(form)
        self.entry_verse.grid(row=9, column=1, sticky="w")
        tk.Label(form, text="Title", bg=BG_COLOR).grid(row=10, column=0, sticky="w")
        self.entry_title = tk.Entry(form, width=40)
        self.entry_title.grid(row=10, column=1, sticky="w")

        # save edits button
        tk.Button(form, text="Save Changes", bg=FRAME_COLOR, fg="white", command=self.save_changes).grid(row=11, column=0, pady=6)
        tk.Button(form, text="Refresh List", command=self.refresh_list).grid(row=11, column=1, pady=6)

        # populate list
        self.refresh_list()

    def refresh_list(self):
        self.listbox.delete(0, tk.END)
        for i, v in enumerate(verses):
            ch = v.get("chapter_number", "?"); vn = v.get("verse_number", "?")
            title = v.get("title") or ""
            self.listbox.insert(tk.END, f"{i}: {ch}:{vn}  {title}")

    def on_select(self, evt=None):
        sel = self.listbox.curselection()
        if not sel:
            return
        i = sel[0]
        v = verses[i]
        self.txt_sanskrit.delete("1.0", tk.END); self.txt_sanskrit.insert(tk.END, v.get("text",""))
        self.txt_translit.delete("1.0", tk.END); self.txt_translit.insert(tk.END, v.get("transliteration",""))
        self.txt_translation.delete("1.0", tk.END); self.txt_translation.insert(tk.END, v.get("translation",""))
        self.txt_meanings.delete("1.0", tk.END); self.txt_meanings.insert(tk.END, v.get("word_meanings",""))
        self.entry_chapter.delete(0, tk.END); self.entry_chapter.insert(0, v.get("chapter_number",""))
        self.entry_verse.delete(0, tk.END); self.entry_verse.insert(0, v.get("verse_number",""))
        self.entry_title.delete(0, tk.END); self.entry_title.insert(0, v.get("title",""))

    def save_changes(self):
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showwarning("Select", "Select a verse to save changes to.")
            return
        i = sel[0]
        v = verses[i]
        v["text"] = self.txt_sanskrit.get("1.0", tk.END).strip()
        v["transliteration"] = self.txt_translit.get("1.0", tk.END).strip()
        v["translation"] = self.txt_translation.get("1.0", tk.END).strip()
        v["word_meanings"] = self.txt_meanings.get("1.0", tk.END).strip()
        # numeric chapter/verse
        try:
            v["chapter_number"] = int(self.entry_chapter.get().strip())
        except Exception:
            v["chapter_number"] = self.entry_chapter.get().strip()
        try:
            v["verse_number"] = int(self.entry_verse.get().strip())
        except Exception:
            v["verse_number"] = self.entry_verse.get().strip()
        v["title"] = self.entry_title.get().strip()
        # update TF-IDF and semantic indices
        build_tfidf_index(verses)
        threading.Thread(target=try_build_semantic_index, daemon=True).start()
        messagebox.showinfo("Saved", "Verse updated in memory. Use Save JSON to persist to disk.")
        self.refresh_list()

    def add_verse(self):
        # create a minimal verse with next id
        new = {
            "chapter_number": simpledialog.askinteger("chapter", "Enter chapter number", initialvalue=1),
            "verse_number": simpledialog.askinteger("verse", "Enter verse number", initialvalue=1),
            "title": simpledialog.askstring("title", "Title / short description"),
            "text": simpledialog.askstring("text", "Devanagari text (paste)"),
            "transliteration": simpledialog.askstring("translit", "Transliteration (IAST or ascii)"),
            "translation": simpledialog.askstring("translation", "English translation / explanation"),
            "word_meanings": simpledialog.askstring("meanings", "Word-meanings (English)")
        }
        verses.append(new)
        build_tfidf_index(verses)
        threading.Thread(target=try_build_semantic_index, daemon=True).start()
        self.refresh_list()

    def delete_verse(self):
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showwarning("Select", "Choose a verse to delete.")
            return
        i = sel[0]
        if messagebox.askyesno("Delete", "Delete selected verse?"):
            verses.pop(i)
            build_tfidf_index(verses)
            threading.Thread(target=try_build_semantic_index, daemon=True).start()
            self.refresh_list()

    def save_json(self):
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files","*.json"),("All files","*.*")])
        if not path:
            return
        ok, err = save_verses_to_path(path)
        if ok:
            messagebox.showinfo("Saved", f"Saved {len(verses)} verses to {path}")
        else:
            messagebox.showerror("Error", f"Failed to save: {err}")

class KrishnaChatWindow:
    def __init__(self, parent):
        self.win = tk.Toplevel(parent)
        self.win.title("Dialogue with Krishna")
        self.win.geometry("700x700")
        self.win.configure(bg=BG_COLOR)
        
        header = tk.Label(self.win, text="Guidance of the Gita — Arjuna & Krishna", font=("Noto Serif Devanagari", 18, "bold"), bg=BG_COLOR, fg=GOLD)
        header.pack(pady=10)
        
        self.chat_history = scrolledtext.ScrolledText(self.win, wrap=tk.WORD, bg=CARD_BG, font=("Georgia", 12))
        self.chat_history.pack(fill="both", expand=True, padx=10, pady=5)
        self.chat_history.tag_config("arjuna", foreground="#2a52be", font=("Georgia", 12, "bold"))
        self.chat_history.tag_config("krishna", foreground="#b76e22", font=("Georgia", 13, "bold"))
        self.chat_history.tag_config("normal", foreground=TEXT_COLOR, font=("Georgia", 12))
        self.chat_history.insert(tk.END, "Krishna: ", "krishna")
        self.chat_history.insert(tk.END, "Welcome, O Arjuna. Speak what troubles your mind.\n\n", "normal")
        self.chat_history.config(state="disabled")
        
        inp_frame = tk.Frame(self.win, bg=BG_COLOR)
        inp_frame.pack(fill="x", padx=10, pady=10)
        
        self.input_box = scrolledtext.ScrolledText(inp_frame, wrap=tk.WORD, height=3, font=("Georgia", 12))
        self.input_box.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.input_box.bind("<Return>", self.send_message_event)
        
        send_btn = tk.Button(inp_frame, text="Ask Krishna", bg=FRAME_COLOR, fg="white", font=("Georgia", 12, "bold"), command=self.send_message)
        send_btn.pack(side="right", fill="y", ipadx=10)
        
    def send_message_event(self, event):
        self.send_message()
        return "break"
        
    def send_message(self):
        user_text = self.input_box.get("1.0", tk.END).strip()
        if not user_text:
            return
            
        self.input_box.delete("1.0", tk.END)
        self.chat_history.config(state="normal")
        self.chat_history.insert(tk.END, "Arjuna (You): ", "arjuna")
        self.chat_history.insert(tk.END, user_text + "\n\n", "normal")
        
        reply = krishna_reply(user_text, top_k=3)
        self.chat_history.insert(tk.END, "Krishna: ", "krishna")
        
        lines = reply.split("\n")
        if "मित्रा" in lines[0] or "मित्र" in lines[0]:
            lines[0] = lines[0].replace("कृष्ण: ", "")
            
        filtered_reply = "\n".join(lines)
        self.chat_history.insert(tk.END, filtered_reply + "\n\n", "normal")
        self.chat_history.yview(tk.END)
        self.chat_history.config(state="disabled")

class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bhagavad Gita — Wisdom Engine")
        self.root.configure(bg=BG_COLOR)
        self.recommended_cache = []  # list of (verse,score)

        header = tk.Label(root, text="ॐ तत् सत्\nBhagavad Gita — Wisdom Engine", font=("Noto Serif Devanagari", 20, "bold"), bg=BG_COLOR, fg=GOLD)
        header.pack(pady=6)

        # input
        tk.Label(root, text="Enter your thoughts (English or Devanagari):", bg=BG_COLOR, fg=TEXT_COLOR).pack(anchor="w", padx=10)
        self.input_box = scrolledtext.ScrolledText(root, height=5, bg=CARD_BG)
        self.input_box.pack(fill="both", padx=10, pady=6)

        # controls
        ctrl = tk.Frame(root, bg=BG_COLOR)
        ctrl.pack(fill="x", padx=10)
        tk.Button(ctrl, text="Analyze", bg=FRAME_COLOR, fg="white", command=self.analyze).pack(side="left", padx=4)
        tk.Button(ctrl, text="Recommend", bg=FRAME_COLOR, fg="white", command=self.recommend).pack(side="left", padx=4)
        tk.Button(ctrl, text="Load Gita JSON", bg=FRAME_COLOR, fg="white", command=self.load_json_dialog).pack(side="left", padx=4)
        tk.Button(ctrl, text="Verse Manager", bg=FRAME_COLOR, fg="white", command=self.open_manager).pack(side="left", padx=4)
        tk.Button(ctrl, text="Krishna Chat", bg=FRAME_COLOR, fg="white", command=self.krishna_chat_prompt).pack(side="left", padx=4)
        tk.Button(ctrl, text="Export Bookmarks", bg=FRAME_COLOR, fg="white", command=self.export_bookmarks).pack(side="left", padx=4)

        # two-column layout for outputs and recommendations
        main = tk.Frame(root, bg=BG_COLOR)
        main.pack(fill="both", expand=True, padx=10, pady=6)

        left = tk.Frame(main, bg=BG_COLOR)
        left.pack(side="left", fill="both", expand=True)

        right = tk.Frame(main, width=320, bg=BG_COLOR)
        right.pack(side="left", fill="y", padx=(10,0))

        # left outputs: tokens, stats, sentiment, recommendations
        tk.Label(left, text="Tokens (token → type):", bg=BG_COLOR).pack(anchor="w")
        self.token_box = scrolledtext.ScrolledText(left, height=6)
        self.token_box.pack(fill="both", pady=4)

        tk.Label(left, text="Token statistics:", bg=BG_COLOR).pack(anchor="w")
        self.stat_box = scrolledtext.ScrolledText(left, height=3)
        self.stat_box.pack(fill="both", pady=4)

        tk.Label(left, text="Sentiment / Mood:", bg=BG_COLOR).pack(anchor="w")
        self.sentiment_box = scrolledtext.ScrolledText(left, height=4)
        self.sentiment_box.pack(fill="both", pady=4)

        tk.Label(left, text="Recommendations:", bg=BG_COLOR).pack(anchor="w")
        # recommendations area
        self.reco_box = scrolledtext.ScrolledText(left, height=10)
        self.reco_box.pack(fill="both", pady=6)
        self.reco_box.bind("<Double-Button-1>", self.on_reco_double)

        # right side: verse list, bookmarks
        tk.Label(right, text="Loaded verses:", bg=BG_COLOR).pack(anchor="w")
        self.verse_listbox = tk.Listbox(right, height=10)
        self.verse_listbox.pack(fill="both", pady=4)
        self.verse_listbox.bind("<Double-Button-1>", self.on_verse_list_double)

        btns = tk.Frame(right, bg=BG_COLOR)
        btns.pack(fill="x", pady=4)
        tk.Button(btns, text="Refresh List", bg=FRAME_COLOR, fg="white", command=self.refresh_verse_list).pack(side="left", padx=2)
        tk.Button(btns, text="Open Selected", bg=FRAME_COLOR, fg="white", command=self.open_selected_from_list).pack(side="left", padx=2)

        tk.Label(right, text="Bookmarks:", bg=BG_COLOR).pack(anchor="w", pady=(8,0))
        self.bookmark_list = tk.Listbox(right, height=6)
        self.bookmark_list.pack(fill="both", pady=4)
        bm_btns = tk.Frame(right, bg=BG_COLOR)
        bm_btns.pack(fill="x")
        tk.Button(bm_btns, text="Bookmark Selected Reco", command=self.bookmark_selected_reco).pack(side="left", padx=2)
        tk.Button(bm_btns, text="Open Bookmark", command=self.open_bookmark).pack(side="left", padx=2)

    def analyze(self):
        txt = self.input_box.get("1.0", tk.END).strip()
        tokens, types = tokenize(txt)
        self.token_box.delete("1.0", tk.END)
        self.token_box.insert(tk.END, "\n".join(f"{t}  →  {ty}" for t,ty in zip(tokens, types)))
        stats = get_token_stats(tokens, types)
        self.stat_box.delete("1.0", tk.END)
        for k,v in stats.items():
            self.stat_box.insert(tk.END, f"{k}: {v}\n")
        label, comp, details = combined_sentiment(txt)
        mood = detect_mood(txt)
        self.sentiment_box.delete("1.0", tk.END)
        self.sentiment_box.insert(tk.END, f"Label: {label}\nCompound: {comp:.3f}\nMood: {mood}\n")
        if details:
            self.sentiment_box.insert(tk.END, "Details summary:\n")
            if "by_sentence" in details:
                for d in details["by_sentence"]:
                    self.sentiment_box.insert(tk.END, f" - \"{d['sentence'][:60]}...\" → {d['compound']:.3f}\n")
            elif "vader" in details:
                self.sentiment_box.insert(tk.END, f"VADER: {details['vader']}\n")

    def recommend(self):
        txt = self.input_box.get("1.0", tk.END).strip()
        if not txt:
            messagebox.showwarning("Empty", "Write something first.")
            return
        if not verses:
            messagebox.showwarning("No verses", "Load verses JSON first.")
            return
        # prefer semantic if available, else TF-IDF
        results = []
        if semantic_model and semantic_embeddings is not None:
            results = recommend_semantic(txt, top_n=8)
        else:
            results = recommend_tfidf(txt, top_n=8)
        self.recommended_cache = results
        self.reco_box.delete("1.0", tk.END)
        for i, (v, score) in enumerate(results):
            ch = v.get("chapter_number","?"); vn = v.get("verse_number","?")
            title = v.get("title","")
            trans_preview = (v.get("translation") or v.get("word_meanings") or "").splitlines()[0][:140]
            self.reco_box.insert(tk.END, f"[{i}] Chapter {ch}:{vn}  {title}  score={score:.3f}\n")
            # show english translation preview when available
            if v.get("translation") or v.get("word_meanings"):
                self.reco_box.insert(tk.END, "    " + trans_preview + "\n\n")
            else:
                # fallback to transliteration or devanagari first line
                preview = (v.get("transliteration") or v.get("text","")).splitlines()[0][:120]
                self.reco_box.insert(tk.END, "    " + preview + "\n\n")
        self.reco_box.insert(tk.END, "Double-click a line to open the reader.\n")

    def on_reco_double(self, event):
        # find clicked line, extract [i]
        index = self.reco_box.index("@%d,%d" % (event.x, event.y))
        line_no = int(index.split(".")[0])
        line_text = self.reco_box.get(f"{line_no}.0", f"{line_no}.end")
        m = re.search(r"\[(\d+)\]", line_text)
        if not m:
            # maybe the previous line contains the index
            prev = line_no-1
            if prev >= 1:
                line_text = self.reco_box.get(f"{prev}.0", f"{prev}.end")
                m = re.search(r"\[(\d+)\]", line_text)
        if m:
            i = int(m.group(1))
            if 0 <= i < len(self.recommended_cache):
                verse, score = self.recommended_cache[i]
                idx = verses.index(verse)
                ReaderWindow(self.root, idx, app_ref=self)

    def bookmark_selected_reco(self):
        # bookmark currently selected recommendation (first selected line)
        sel_text = None
        try:
            sel = self.reco_box.get("sel.first", "sel.last")
            sel_text = sel.splitlines()[0]
        except Exception:
            # no selection - take first recommendation
            if self.recommended_cache:
                v,score = self.recommended_cache[0]
                self.bookmark_list.insert(tk.END, f"{v.get('chapter_number')}:{v.get('verse_number')} { (v.get('translation') or '')[:40] }")
                return
            else:
                messagebox.showwarning("No selection", "Select a recommendation or generate recommendations first.")
                return
        m = re.search(r"\[(\d+)\]", sel_text)
        if m:
            i = int(m.group(1))
            if 0 <= i < len(self.recommended_cache):
                v,score = self.recommended_cache[i]
                self.bookmark_list.insert(tk.END, f"{v.get('chapter_number')}:{v.get('verse_number')} { (v.get('translation') or '')[:40] }")
                messagebox.showinfo("Bookmarked", "Added to bookmarks.")

    def open_bookmark(self):
        sel = self.bookmark_list.curselection()
        if not sel:
            messagebox.showwarning("Select", "Choose a bookmark to open.")
            return
        i = sel[0]
        entry = self.bookmark_list.get(i)
        # parse chapter:verse at start
        m = re.match(r"^\s*(\d+):(\d+)", entry)
        if m:
            ch, vn = int(m.group(1)), int(m.group(2))
            # locate
            for idx, v in enumerate(verses):
                if int(v.get("chapter_number", -1)) == ch and int(v.get("verse_number", -1)) == vn:
                    ReaderWindow(self.root, idx, app_ref=self)
                    return
        messagebox.showinfo("Not found", "Could not find verse from bookmark in current dataset.")

    def export_bookmarks(self):
        if self.bookmark_list.size() == 0:
            messagebox.showinfo("No bookmarks", "No bookmarks to export.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files","*.txt")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf8") as f:
                for i in range(self.bookmark_list.size()):
                    f.write(self.bookmark_list.get(i) + "\n")
            messagebox.showinfo("Exported", f"Bookmarks exported to {path}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))

    def load_json_dialog(self):
        path = filedialog.askopenfilename(filetypes=[("JSON files","*.json"),("All files","*.*")])
        if not path:
            return
        ok, info = load_verses_from_path(path)
        if not ok:
            messagebox.showerror("Load error", str(info))
        else:
            messagebox.showinfo("Loaded", f"{info} verses loaded.")
            self.refresh_verse_list()

    def refresh_verse_list(self):

        self.verse_listbox.delete(0, tk.END)

        for v in verses:

            ch = v.get("chapter_number","?")
            vn = v.get("verse_number","?")
            title = v.get("title","")

            translation = (v.get("translation") or v.get("word_meanings") or "").strip()

            if translation:
                preview = translation.splitlines()[0][:50]
            else:
                preview = ""

            self.verse_listbox.insert(
                tk.END,
                f"{ch}:{vn} — {preview} {title}"
            )

    def on_verse_list_double(self, event):
        sel = self.verse_listbox.curselection()
        if not sel:
            return
        i = sel[0]
        ReaderWindow(self.root, i, app_ref=self)

    def open_selected_from_list(self):
        sel = self.verse_listbox.curselection()
        if not sel:
            messagebox.showwarning("Select", "Choose a verse from list.")
            return
        i = sel[0]
        ReaderWindow(self.root, i, app_ref=self)

    def open_manager(self):
        VerseManagerWindow(self.root, app_ref=self)

    def krishna_chat_prompt(self):
        KrishnaChatWindow(self.root)
    # End MainApp

# -------------------------------------------------------------------------
# Bootstrapping & main
# -------------------------------------------------------------------------
def main():
    root = tk.Tk()
    app = MainApp(root)

    # Optionally try to auto-load a bundled verses.json next to executable
    bundled_paths = [
        resource_path("verse.json"),
        os.path.join(os.path.dirname(__file__), "verse.json"),
        "verse.json"
    ]
    for p in bundled_paths:
        if os.path.exists(p):
            ok, info = load_verses_from_path(p)
            if ok:
                app.refresh_verse_list()
            break

    root.mainloop()

if __name__ == "__main__":
    main()
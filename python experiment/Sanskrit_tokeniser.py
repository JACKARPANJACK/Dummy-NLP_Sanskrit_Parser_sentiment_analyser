#!/usr/bin/env python3
"""
sanskrit_tokenizer.py
A simple Sanskrit tokeniser with Unicode normalization, diacritic handling,
basic tokenization, and a heuristic Sandhi splitter.
Features:
- Unicode normalization (NFC/NFKC)
- Optional diacritic stripping (so 'kṛṣṇa' -> 'krshna' / 'krishna' heuristics)
- Basic tokenization (words, numbers, punctuation)
- Simple heuristic Sandhi splitter (can be replaced with lexicon-based splitter)
- CLI example usage
"""

import re
import unicodedata
from typing import List, Tuple



# --- Configuration / Regexes -------------------------------------------------

# Basic "word" regex allowing diacritics and ascii letters, plus apostrophes/hyphens
WORD_RE = re.compile(r"[A-Za-z\u0100-\u024F\u1E00-\u1EFF]+(?:[-'][A-Za-z\u0100-\u024F\u1E00-\u1EFF]+)*")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
PUNCT_RE = re.compile(r"[^\s\w\u0100-\u024F\u1E00-\u1EFF]")


# --- Normalization / Diacritic handling -------------------------------------

def normalize_text(text: str, form: str = "NFC") -> str:
    """
    Normalize unicode string (NFC by default).
    Use NFKC if you want compatibility decomposition (not usually necessary).
    """
    return unicodedata.normalize(form, text)


def strip_diacritics(text: str) -> str:
    """
    Strip diacritics by decomposing and removing combining marks.
    Example: 'kṛṣṇa' -> 'krshna' (note: ṛ -> r + combining dot; 'ṣ' -> 's' with dot below).
    This will not transliterate ṛ -> 'ri' or 'ṛ' -> 'r' with vowel changes — that's a heuristic.
    """
    nfkd = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    # Some combined characters like 'ś' may decompose to 's'+'´' so this is handled above.
    return unicodedata.normalize("NFC", stripped)


def iast_to_simple_ascii(iast: str) -> str:
    """
    Optionally map special letters to ASCII approximations (small set).
    This is a short manual mapping useful for readability/search. Expand as needed.
    e.g. 'ā'->'a', 'ī'->'i', 'ū'->'u', 'ṛ'->'ri' or 'r' (choice).
    We'll choose simple base-letter approximations (ā->a, ṛ->r).
    """
    mapping = {
        "ā": "a", "ī": "i", "ū": "u", "ṛ": "r", "ṝ": "r",
        "ḷ": "l", "ḹ": "l",
        "ṅ": "n", "ñ": "n", "ṇ": "n", "ṣ": "s", "ś": "s", "ḥ": "h",
        "ḍ": "d", "ṭ": "t", "ḷ": "l", "ṁ": "m",
        "Ā": "A", "Ī": "I", "Ū": "U", "Ṛ": "R", "Ṝ": "R",
        "Ṅ": "N", "Ñ": "N", "Ṇ": "N", "Ṣ": "S", "Ś": "S", "Ḥ": "H",
        "Ḍ": "D", "Ṭ": "T", "Ḷ": "L", "Ṁ": "M",
    }
    out = []
    for ch in iast:
        out.append(mapping.get(ch, ch))
    return "".join(out)


# --- Tokenizers --------------------------------------------------------------

def basic_tokenize(text: str, normalize: bool = True, strip_diac: bool = False,
                   iast_ascii: bool = False) -> List[str]:
    """
    Tokenize text into tokens: words (including diacritics), numbers, punctuation.
    Options:
      - normalize: run Unicode normalization
      - strip_diac: remove diacritics (combine with iast_ascii for ASCII approximations)
      - iast_ascii: convert common IAST letters to ASCII approximations after stripping diacritics
    """
    if normalize:
        text = normalize_text(text)

    if strip_diac:
        text = strip_diacritics(text)

    if iast_ascii:
        text = iast_to_simple_ascii(text)

    tokens: List[str] = []
    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue

        # number
        m = NUMBER_RE.match(text, i)
        if m:
            tokens.append(m.group(0))
            i = m.end()
            continue

        # word (letters + diacritics range included)
        m = WORD_RE.match(text, i)
        if m:
            tokens.append(m.group(0))
            i = m.end()
            continue

        # punctuation
        m = PUNCT_RE.match(text, i)
        if m:
            tokens.append(m.group(0))
            i = m.end()
            continue

        # fallback: take single char
        tokens.append(text[i])
        i += 1

    return tokens


# --- Simple Sandhi splitting heuristic --------------------------------------

VOWELS = set("aeiouāīūṛṝḷAEIOUĀĪŪṚṜḶ")
CONSONANTS = set("bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ" +
                 "ṅñṇṣśḍṭḷḍṬṆ")  # partial extended set

def simple_sandhi_splits(token: str, min_part_len: int = 2) -> List[Tuple[str, str]]:
    """
    Heuristic splitter: try to split a token into (left, right) where the join looks like a
    common vowel-vowel sandhi or consonant+vowel boundary.
    This is NOT linguistically complete. It returns candidate splits (could be empty).
    If you have a lexicon, replace scoring with checks against known words.
    """
    token = token.strip()
    n = len(token)
    candidates = []

    # don't attempt trivial/small tokens
    if n < (min_part_len * 2):
        return candidates

    # try splits across the token
    for i in range(min_part_len, n - min_part_len + 1):
        L = token[:i]
        R = token[i:]

        # heuristic 1: vowel + vowel boundary (vowel sandhi likely)
        if L[-1] in VOWELS and R[0] in VOWELS:
            candidates.append((L, R))
            continue

        # heuristic 2: consonant cluster where last char of left is consonant and right starts with vowel
        if L[-1] not in VOWELS and R[0] in VOWELS:
            candidates.append((L, R))
            continue

        # heuristic 3: visarga-like 'ḥ' at end of left or 's' assimilation
        if L.endswith("ḥ") or L.endswith("h") and R[0] in VOWELS:
            candidates.append((L, R))
            continue

    # optional: remove duplicates and sort by closeness to center
    def score(split):
        L, R = split
        return abs(len(L) - len(R))
    candidates = sorted(set(candidates), key=score)
    return candidates


# --- High-level pipeline ----------------------------------------------------

def pipeline_tokenize(text: str,
                      normalize: bool = True,
                      strip_diacritics_flag: bool = False,
                      iast_ascii_flag: bool = False,
                      sandhi_split: bool = False) -> List[str]:
    """
    Full pipeline: tokenization + optional sandhi splitting that expands tokens list.
    If sandhi_split=True, uses simple_sandhi_splits and picks the first candidate (best heuristic).
    """
    toks = basic_tokenize(text, normalize=normalize,
                          strip_diac=strip_diacritics_flag,
                          iast_ascii=iast_ascii_flag)

    if not sandhi_split:
        return toks

    expanded = []
    for t in toks:
        if re.fullmatch(WORD_RE, t):
            splits = simple_sandhi_splits(t)
            if splits:
                # pick the first (best) candidate for now; could return all candidates
                left, right = splits[0]
                expanded.extend([left, right])
                continue
        expanded.append(t)

    return expanded


# --- CLI / Example usage ----------------------------------------------------

if __name__ == "__main__":
    examples = [
        "aham vande gurūn. kṛṣṇaḥ paṭhan.",
        "namaste! this is a mixed english-sanskrit sentence: dharma, karma, mokṣa.",
        "rāmaś candramāsaḥ -> rāma ś candramāsaḥ (example with visarga).",
        "sangeetam and śiva-līlā at the temple.",
        "tat tvam asi",
    ]

    print("=== BASIC TOKENIZATION ===")
    for ex in examples:
        print("\nInput:", ex)
        print("Tokens:", basic_tokenize(ex))
        print("Tokens (strip diacritics):", basic_tokenize(ex, strip_diac=True))
        print("Tokens (ascii iast):", basic_tokenize(ex, strip_diac=True, iast_ascii=True))

    print("\n=== SANDHI SPLIT HEURISTICS ===")
    test_tokens = ["rāmachandra", "kṛṣṇacandra", "guruḥkṛṣṇa", "ahamasti"]
    for t in test_tokens:
        print(f"{t} -> splits: {simple_sandhi_splits(t)}")

    # small interactive demo
    s = "kṛṣṇaḥrāma"
    print("\nPipeline example:")
    print("Input:", s)
    print("Tokens:", pipeline_tokenize(s, strip_diacritics_flag=True, sandhi_split=True))


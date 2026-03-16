# 🕉 Bhagavad Gita Wisdom Engine

An interactive **Bhagavad Gita exploration tool** that combines **Natural Language Processing, sentiment analysis, semantic search, and Sanskrit-aware tokenization** to recommend verses from the Bhagavad Gita based on a user's thoughts or emotional state.

This project was built as an experimental **NLP + Sanskrit text processing tool** and demonstrates how traditional scriptures can be explored using modern AI/NLP techniques.

---

# ✨ Features

### 📜 Scripture Reader
- Two-panel reader:
  - Left → Sanskrit (Devanagari)
  - Right → English translation + commentary
- Verse navigation
  - Previous / Next verse
  - Jump to chapter and verse
- Adjustable font size
- Copy translation
- Save explanations / notes

---

### 🧠 NLP Analysis
The engine analyzes user input using:

- Tokenization
- Sentiment analysis
- Mood detection
- Sanskrit + English language handling

Supported token types:
- English words
- Devanagari words
- Numbers
- Punctuation

---

### 📊 Sentiment Engine

Sentiment detection combines:

- **NLTK VADER sentiment analysis**
- **Custom Hindi lexicon boost**
- **Devanagari transliteration**

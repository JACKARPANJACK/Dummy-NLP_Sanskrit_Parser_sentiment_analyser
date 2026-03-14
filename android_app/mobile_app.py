import os
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window

# -------------------------------------------------------------------------
# Core Logic (Stripped of Tkinter/NLTK for mobile simplicity in this demo)
# -------------------------------------------------------------------------
verses = []
tf_vectorizer = None
tf_matrix = None

def load_verses_from_path(path):
    global verses, tf_vectorizer, tf_matrix
    try:
        with open(path, "r", encoding="utf8") as f:
            data = json.load(f)
            if isinstance(data, dict) and data.get("verses"):
                verses = data["verses"]
            elif isinstance(data, list):
                verses = data
        
        # Load translation.json if available
        base_dir = os.path.dirname(path)
        trans_path = os.path.join(base_dir, "translation.json")
        if not os.path.exists(trans_path):
            trans_path = "translation.json"
            
        if os.path.exists(trans_path):
            try:
                with open(trans_path, "r", encoding="utf8") as tf:
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
                print(f"Failed to load translations: {e}")

        # Load commentary.json if available
        comm_path = os.path.join(base_dir, "commentary.json")
        if not os.path.exists(comm_path):
            comm_path = "commentary.json"

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
                        text_entry = f"[{lang.upper()}]\n{desc}"
                        if "Shankaracharya" in author:
                            comm_map[vid]["shankara"].append(text_entry)
                        elif "Ramanujacharya" in author:
                            comm_map[vid]["ramanuja"].append(text_entry)
                        elif "Madhavacharya" in author:
                            comm_map[vid]["madhva"].append(text_entry)

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
                print(f"Failed to load commentaries: {e}")

        # Build TF-IDF
        corpus = []
        for v in verses:
            text_parts = []
            if v.get("translation"): text_parts.append(v["translation"])
            if v.get("word_meanings"): text_parts.append(v["word_meanings"])
            if v.get("commentary_shankara"): text_parts.append(v["commentary_shankara"])
            if v.get("commentary_ramanuja"): text_parts.append(v["commentary_ramanuja"])
            if v.get("commentary_madhva"): text_parts.append(v["commentary_madhva"])
        return True, len(verses)
    except Exception as e:
        return False, str(e)

def recommend_tfidf(query, top_n=3):
    global tf_vectorizer, tf_matrix
    if tf_vectorizer is None or tf_matrix is None:
        return []
    
    q_vec = tf_vectorizer.transform([query.lower()])
    similarities = cosine_similarity(q_vec, tf_matrix)[0]
    idxs = similarities.argsort()[::-1][:top_n]
    
    return [(verses[i], float(similarities[i])) for i in idxs]

def krishna_reply(user_text):
    recs = recommend_tfidf(user_text, top_n=2)
    parts = ["Krishna: Welcome, O friend. Listen to these words:\n"]
    for (v, _) in recs:
        ch = v.get("chapter_number", "?")
        vn = v.get("verse_number", "?")
        trans = v.get("translation") or v.get("word_meanings") or ""
        short = trans.splitlines()[0][:150]
        parts.append(f"- Gita {ch}.{vn}: {short}")
    
    if not recs:
        parts.append("I couldn't find a matching verse. Please reflect and speak again.")
    return "\n\n".join(parts)


# -------------------------------------------------------------------------
# Kivy App Interface
# -------------------------------------------------------------------------
class GitaAppLayout(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', padding=10, spacing=10, **kwargs)
        Window.clearcolor = (0.96, 0.90, 0.78, 1) # BG_COLOR #f5e6c8 approx
        
        # Header
        self.header = Label(
            text="Bhagavad Gita — Mobile Engine",
            size_hint=(1, 0.1),
            color=(0.83, 0.68, 0.21, 1), # GOLD approx
            font_size='20sp',
            bold=True
        )
        self.add_widget(self.header)
        
        import os
        
        # UI Structure
        self.tabs = BoxLayout(size_hint=(1, 0.08))
        self.btn_main = Button(text="Search", background_color=(0.76, 0.54, 0.17, 1))
        self.btn_chat = Button(text="Krishna Chat", background_color=(0.76, 0.54, 0.17, 1))
        self.btn_bookmarks = Button(text="Bookmarks", background_color=(0.76, 0.54, 0.17, 1))
        
        self.tabs.add_widget(self.btn_main)
        self.tabs.add_widget(self.btn_chat)
        self.tabs.add_widget(self.btn_bookmarks)
        self.add_widget(self.tabs)
        
        # View Management
        self.container = BoxLayout(orientation='vertical')
        self.add_widget(self.container)
        
        # Load Data
        self.bookmarks = []
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python experiment", ".venv", "Scripts", "verse.json")
        ok, msg = load_verses_from_path(json_path)
        self.dataset_info = f"Loaded {msg} verses" if ok else "Failed to load"

        # Initialize Main View
        self.show_main_view()

        # Bind tabs
        self.btn_main.bind(on_press=self.show_main_view)
        self.btn_chat.bind(on_press=self.show_chat_view)
        self.btn_bookmarks.bind(on_press=self.show_bookmarks_view)

    def show_main_view(self, *args):
        self.container.clear_widgets()
        
        # Top Stats Display
        self.scroll = ScrollView(size_hint=(1, 0.6))
        self.output_label = Label(
            text=self.dataset_info + "\nAsk a query to see recommendations, stats & NLP sentiment.",
            size_hint_y=None,
            color=(0.23, 0.16, 0.10, 1),
            text_size=(Window.width - 40, None),
            markup=True
        )
        self.output_label.bind(texture_size=self.output_label.setter('size'))
        self.scroll.add_widget(self.output_label)
        self.container.add_widget(self.scroll)
        
        # Search Box
        self.input_box = TextInput(size_hint=(1, 0.15), hint_text="Search thoughts, Devanagari, English...")
        self.container.add_widget(self.input_box)
        
        # Actions
        btn_layout = BoxLayout(size_hint=(1, 0.15))
        btn_search = Button(text="Analyze & Recommend", background_color=(0.76, 0.54, 0.17, 1))
        btn_search.bind(on_press=self.do_recommend)
        btn_bm = Button(text="Bookmark Top", background_color=(0.5, 0.8, 0.5, 1))
        btn_bm.bind(on_press=self.bookmark_top_reco)
        
        btn_layout.add_widget(btn_search)
        btn_layout.add_widget(btn_bm)
        self.container.add_widget(btn_layout)
        
    def show_chat_view(self, *args):
        self.container.clear_widgets()
        self.chat_scroll = ScrollView(size_hint=(1, 0.7))
        self.chat_history_lbl = Label(text="Arjuna (You): Who am I?\n\nKrishna: You are a divine consciousness.\n\n", size_hint_y=None, color=(0.2, 0.2, 0.2, 1), markup=True, text_size=(Window.width - 40, None))
        self.chat_history_lbl.bind(texture_size=self.chat_history_lbl.setter('size'))
        self.chat_scroll.add_widget(self.chat_history_lbl)
        
        self.container.add_widget(self.chat_scroll)
        
        # Input Box
        self.chat_input = TextInput(size_hint=(1, 0.15), hint_text="Ask Krishna...")
        self.container.add_widget(self.chat_input)
        
        btn_ask = Button(text="Ask Krishna", size_hint=(1, 0.15), background_color=(0.76, 0.54, 0.17, 1))
        btn_ask.bind(on_press=self.do_chat)
        self.container.add_widget(btn_ask)
        
    def show_bookmarks_view(self, *args):
        self.container.clear_widgets()
        scroll = ScrollView()
        text = "[b]Your Saved Bookmarks[/b]\n\n"
        if not self.bookmarks:
            text += "No bookmarks yet. Go to Search & save some!"
        for idx, bm in enumerate(self.bookmarks):
            text += f"{idx+1}. Gita {bm}\n"
            
        lbl = Label(text=text, size_hint_y=None, color=(0.2, 0.2, 0.2, 1), markup=True, text_size=(Window.width - 40, None))
        lbl.bind(texture_size=lbl.setter('size'))
        scroll.add_widget(lbl)
        self.container.add_widget(scroll)

    def do_recommend(self, instance):
        query = self.input_box.text.strip()
        if not query: return
        recs = recommend_tfidf(query)
        self.last_reco = recs
        
        # Fake Sentiments & tokens for mobile layout representation
        tokens = query.split()
        res_text = f"[b]NLP Analysis:[/b]\nTokens: {len(tokens)}\nWords: {tokens}\nSentiment: Neutral (simulated)\n\n"
        
        res_text += f"[b]Recommendations for: '{query}'[/b]\n\n"
        for v, s in recs:
            res_text += f"[color=b76e22]Gita {v.get('chapter_number')}:{v.get('verse_number')}[/color] (Score: {s:.2f})\n"
            res_text += f"{(v.get('translation') or v.get('word_meanings') or '')[:150]}...\n\n"
            
        self.output_label.text = res_text
        self.input_box.text = ""
        
    def bookmark_top_reco(self, instance):
        if hasattr(self, 'last_reco') and self.last_reco:
            v, _ = self.last_reco[0]
            self.bookmarks.append(f"{v.get('chapter_number')}:{v.get('verse_number')}")
            self.output_label.text = f"[color=008800]Bookmarked top result: Gita {v.get('chapter_number')}:{v.get('verse_number')}[/color]\n\n" + self.output_label.text
        
    def do_chat(self, instance):
        query = self.chat_input.text.strip()
        if not query: return
        reply = krishna_reply(query)
        self.chat_history_lbl.text += f"[color=2a52be][b]Arjuna (You):[/b][/color] {query}\n\n[color=b76e22][b]{reply}[/b][/color]\n\n"
        self.chat_input.text = ""

class GitaApp(App):
    def build(self):
        return GitaAppLayout()

if __name__ == '__main__':
    GitaApp().run()
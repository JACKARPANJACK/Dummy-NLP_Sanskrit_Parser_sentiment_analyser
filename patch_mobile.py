import sys
import os

with open(r'android_app\mobile_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

if 'import pickle' not in content:
    content = content.replace('import json', 'import json\nimport pickle\nimport random')

new_load = r'''def load_verses_from_path(path):
    global verses, tf_vectorizer, tf_matrix
    try:
        base_dir = os.path.dirname(path)
        cache_path = os.path.join(base_dir, "verses_cache.pkl")
        
        # Try loading from cache first
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    cache_data = pickle.load(f)
                verses = cache_data.get("verses", [])
                tf_vectorizer = cache_data.get("tf_vectorizer")
                tf_matrix = cache_data.get("tf_matrix")
                print("Loaded verses and TF-IDF from cache successfully!")
                return True, len(verses)
            except Exception as e:
                print("Notice: Cache load failed:", e)

        with open(path, "r", encoding="utf8") as f:
            data = json.load(f)'''

content = content.replace(r'''def load_verses_from_path(path):
    global verses, tf_vectorizer, tf_matrix
    try:
        with open(path, "r", encoding="utf8") as f:
            data = json.load(f)''', new_load)

old_end = r'''        if corpus:
            tf_vectorizer = TfidfVectorizer(stop_words="english")
            tf_matrix = tf_vectorizer.fit_transform(corpus)
        return True, len(verses)
    except Exception as e:
        return False, str(e)'''

new_end = r'''        if corpus:
            tf_vectorizer = TfidfVectorizer(stop_words="english")
            tf_matrix = tf_vectorizer.fit_transform(corpus)
            
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({
                    "verses": verses,
                    "tf_vectorizer": tf_vectorizer,
                    "tf_matrix": tf_matrix
                }, f)
            print("Successfully cached verses and internal matrix.")
        except Exception as e:
            print("Warning: Failed to save cache:", e)
            
        return True, len(verses)
    except Exception as e:
        return False, str(e)'''

content = content.replace(old_end, new_end)


start = content.find('def krishna_reply')
end = content.find('# -------------------------------------------------------------------------', start)

new_reply = r'''def krishna_reply(user_text):
    recs = recommend_tfidf(user_text, top_n=2)
    parts = []
    
    intros = [
        "कृष्ण: मित्र / O friend, reflect and listen. The truth is eternal. (友よ、反省して聞いてください。)",
        "Krishna: Mein lieber Freund (My dear friend), do not let your mind be clouded by illusions. Here is what you must understand:",
        "कृष्ण: 友よ (Tomo yo - Friend), the material world is fleeting. Listen to the wisdom of the ages:",
        "Krishna: Seien Sie standhaft! (Be steadfast!) O Arjuna, steady your mind and hear these words:",
        "कृष्ण: पार्थ (O Partha), in times of doubt, true knowledge is the only refuge. 聞いてください (Listen):",
        "Krishna: Der Geist ist unruhig, aber das Wissen bringt Frieden. (The mind is restless, but wisdom brings peace). O friend, observe:"
    ]
    
    parts.append(random.choice(intros) + "\n")
    
    for (v, _) in recs:
        ch = v.get("chapter_number", "?")
        vn = v.get("verse_number", "?")
        trans = v.get("translation") or v.get("word_meanings") or ""
        short = trans.splitlines()[0][:150]
        parts.append(f"✦ Gita {ch}.{vn}: {short}")

    if recs:
        parts.append("\n[कृष्ण का मार्गदर्शन / Lord's Guidance]")
        lower_t = user_text.lower()
        if "fear" in lower_t or "sad" in lower_t or "die" in lower_t:
            parts.append("Why this fear? Furcht ist nur eine Illusion. \nThe soul is never born, nor does it die. 恐れないでください (Do not fear).")
        elif "duty" in lower_t or "work" in lower_t or "action" in lower_t:
            parts.append("Tue deine Pflicht ohne Anhaftung (Do your duty without attachment). \n結果に執着せず、義務を果たす (Fulfill your duty without clinging to the results).")
        else:
            parts.append("Wahre Weisheit kommt von innen (True wisdom comes from within). \n心の中で瞑想しなさい (Meditate within your heart).")
    else:
        parts.append("I couldn't find a matching verse. 落ち着いて (Stay calm), and seek your answer within, oder frag mich auf andere Weise.")
        
    return "\n".join(parts)

'''

content = content[:start] + new_reply + content[end:]

with open(r'android_app\mobile_app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Mobile app patched")

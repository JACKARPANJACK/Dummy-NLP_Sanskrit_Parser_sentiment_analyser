import sys
import os

with open(r'python experiment\Main\Scripts\Sanskrit_tokeniser.py', 'r', encoding='utf-8') as f:
    content = f.read()

start = content.find('def krishna_reply')
end = content.find('class ReaderWindow:')

new_func = r'''def krishna_reply(user_text: str, top_k=3):
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
  class ReaderWindow:'''

new_content = content[:start] + new_func + content[end+19:]

with open(r'python experiment\Main\Scripts\Sanskrit_tokeniser.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Applied patch")

import re

with open(r'python experiment\Main\Scripts\Sanskrit_tokeniser.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('  # -------------------------------------------------------------------------', '# -------------------------------------------------------------------------')
text = text.replace('  class ReaderWindow:', 'class ReaderWindow:')
text = text.replace('  # GUI application classes', '# GUI application classes')

with open(r'python experiment\Main\Scripts\Sanskrit_tokeniser.py', 'w', encoding='utf-8') as f:
    f.write(text)


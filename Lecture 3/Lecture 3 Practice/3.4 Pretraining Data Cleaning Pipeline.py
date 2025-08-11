from datasketch import MinHash, MinHashLSH

def minhash_deduplication(texts, threshold=0.7):
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    unique_texts = []
    for i, doc in enumerate(texts):
        m = MinHash(num_perm=128)
        for word in set(doc.split()):
            m.update(word.encode('utf8'))
        if not lsh.query(m):
            lsh.insert(f"doc{i}", m)
            unique_texts.append(doc)
    return unique_texts

from langdetect import detect
from bs4 import BeautifulSoup

def clean_html_and_filter_lang(texts, lang='en'):
    filtered = []
    for txt in texts:
        txt = BeautifulSoup(txt, 'html.parser').get_text()
        try:
            if detect(txt.strip()) == lang:
                filtered.append(txt.strip())
        except:
            continue
    return filtered

import re

def strip_pii(text):
    text = re.sub(r'[\w\.-]+@[\w\.-]+', '[EMAIL]', text)
    text = re.sub(r'\b\d{12,19}\b', '[CREDIT_CARD]', text)
    text = re.sub(r'\b(?:\d{3}-){2}\d{4}\b', '[PHONE]', text)
    return text
     
import re
from collections import Counter

def remove_repetitive_ngrams(text, n=3, threshold=3):
    words = text.split()
    ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

    counts = Counter(ngrams)
    repetitive = [ngram for ngram, count in counts.items() if count >= threshold]

    for phrase in repetitive:
        # regex-safe version of the phrase
        escaped_phrase = re.escape(phrase)
        # match the phrase repeated 2+ times with optional whitespace
        text = re.sub(rf'(?:{escaped_phrase}\s*){{{threshold},}}', phrase + ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text

import pandas as pd
fake_texts = pd.read_csv("D:/Inference Ai Stuff/Lecture 3/Lecture 3 Practice/test_data/data/Fake_Pretraining_Texts.csv")
raw_dataset = fake_texts["Raw Text"]
#print(raw_dataset)

# Step 1: Remove HTML + Language Filter
step1 = clean_html_and_filter_lang(raw_dataset)
#display(step1)

# Step 2: Deduplicate Paragraphs
step2 = minhash_deduplication(step1)
#display(step2)

# Step 3: Strip PII
step3 = [strip_pii(t) for t in step2]
#display(step3)

# Step 4: Remove Repetitive N-grams
cleaned_data = [remove_repetitive_ngrams(t) for t in step3]
print(cleaned_data)
    
# Done!
print("✅ Cleaned dataset sample:")
for idx, text in enumerate(cleaned_data):
    print(f"--- Article {idx + 1} ---")
    print(text)
from bad_dict import get_bad_word_dict
import re
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from collections import OrderedDict

data_path = 'data'
train = 'train.csv'
test = 'test.csv'
train_raw = pd.read_csv(os.path.join(data_path, train))
raw_value = train_raw['comment_text'].fillna("_na_").values


def text_parse(text, remove_stopwords=False, stem_words=False):
    wiki_reg=r'https?://en.wikipedia.org/[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]'
    url_reg=r'https?://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]'
    ip_reg='\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    WIKI_LINK=' WIKILINKREPLACER '
    URL_LINK=' URLLINKREPLACER '
    IP_LINK=' IPLINKREPLACER '
    #clear link
    c = re.findall(wiki_reg, text)
    for u in c:
        text = text.replace(u, WIKI_LINK)
    c = re.findall(url_reg, text)
    for u in c:
        text = text.replace(u, WIKI_LINK)
    c = re.findall(wiki_reg, text)
    for u in c:
        text = text.replace(u, URL_LINK)
    c = re.findall(ip_reg, text)
    for u in c:
        text = text.replace(u, IP_LINK)

    bad_word_dict = get_bad_word_dict()
    # Regex to remove all Non-Alpha Numeric and space
    special_character_removal = re.compile(r'[^A-Za-z\d!?*\'_ ]', re.IGNORECASE)
    # regex to replace all numerics
    replace_numbers = re.compile(r'\b\d+\b', re.IGNORECASE)
    text = text.lower().split()
    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    text = " ".join(text)
    # Remove Special Characters
    text = special_character_removal.sub(' ', text)
    found_dict = {k:False for k in bad_word_dict.keys()}
    for k,v in bad_word_dict.items():
        if text.find(k) >= 0:
           found_dict[k]=True
    return found_dict

bad_word_dict = get_bad_word_dict()
appeared = {k:False for k in bad_word_dict.keys()}
for l in tqdm(raw_value):
     status = text_parse(l)
     for k, v in status.items():
         if v:
             appeared[k]=True
cleaned_dict = {}
for k, v in appeared.items():
    if v:
        cleaned_dict[k] = bad_word_dict[k]
cleaned_dict = OrderedDict(sorted(cleaned_dict.items(), key=lambda t: t[1]))

with open('cleaned_badwords.list', 'w') as f:
    for k, v in cleaned_dict.items():
        if k == v:
            continue
        f.write(k+','+ v +'\n')

import os
import numpy as np
import pandas as pd
from rake_nltk import Rake
from bad_dict import get_bad_word_dict

def rake_parse(line):
    r = Rake()
    r.extract_keywords_from_text(line)
    word_combines = r.get_ranked_phrases()
    word_combines = [k for k in word_combines if len(k.split()) > 1]
    # filter out bad word combines
    bad_word_dict = get_bad_word_dict()
    word_replacer = {}
    for k in word_combines:
        if any(map(lambda x : k.find(x) >= 0, bad_word_dict.values())):
            continue
        word_replacer[k] = '_'.join(k.split())

    for k,v in word_replacer.items():
        line = line.replace(k,v)
    return line

if __name__ == '__main__':
    from preprocess import text_parse
    data_path = 'data'
    train = 'train.csv'
    test = 'test.csv'
    train_raw = pd.read_csv(os.path.join(data_path, train))
    raw_value = train_raw['comment_text'].fillna("_na_").values
    a = raw_value[100]
    print a
    a = text_parse(a)
    print a


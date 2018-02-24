import os
import re
import numpy as np
import pandas as pd
from keras.preprocessing import text, sequence
import config
from rake_nltk import Rake
from bad_dict import get_bad_word_dict

def get_raw_data(path):
    data = pd.read_csv(path)
    process_data = get_data(data)
    data['comment_text'] = process_data
    return data

def get_data(raw_data):
    raw_value = raw_data['comment_text'].fillna("_na_").values
    processed_data = [text_parse(v) for v in raw_value]
    return processed_data 

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
    replace_numbers = re.compile(r'\d+', re.IGNORECASE)
    text = text.lower().split()
    # Optionally, remove stop words

    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    text = " ".join(text)
    # Remove Special Characters
    text = special_character_removal.sub(' ', text)
    for k,v in bad_word_dict.items():
        bad_reg = re.compile('[!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n ]'+ re.escape(k) +'[!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n ]')
        text = bad_reg.sub(' '+ v +' ', text)
    # Replace Numbers
    text = replace_numbers.sub('NUMBER_REPLACER', text)
    text =text.split()
    text = " ".join(text)

    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    #  r = Rake()
    # r.extract_keywords_from_text(text)
    # print r.get_ranked_phrases()

    return text

def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    # Convert words to lower case and split them
    wiki_reg=r'https?://en.wikipedia.org/[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]'
    url_reg=r'https?://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]'
    ip_reg='\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    WIKI_LINK=' WIKI_LINK '
    URL_LINK=' URL_LINK '
    IP_LINK=' IP_LINK '
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

    # Regex to remove all Non-Alpha Numeric and space
    special_character_removal = re.compile(r'[^A-Za-z\d!?*\' ]', re.IGNORECASE)
    # regex to replace all numerics
    replace_numbers = re.compile(r'\d+', re.IGNORECASE)

    # text = text.lower().split()
    text = text.split()
    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    text = " ".join(text)
    # Remove Special Characters
    text = special_character_removal.sub('', text)
    # Replace Numbers
    text = replace_numbers.sub('NUMBERREPLACER', text)
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    # Return a list of words
    return (text)


def get_label(raw_data):
    labels = ['toxic', 'severe_toxic',
              'obscene', 'threat',
              'insult', 'identity_hate']
    return raw_data[labels].values

def get_id(raw_data):
    return raw_data['id'].values

def process_data(train_data, test_data):
    # tokenizer = text.Tokenizer(num_words=config.MAX_WORDS, filters='"#$%&()*+,-./:;<=>@[\\]^_`\'{|}~\t\n', lower=False)
    # tokenizer = text.Tokenizer(num_words=config.MAX_WORDS, filters='-=&\t\n()/\\.#:<>"', lower=False)
    tokenizer = text.Tokenizer(num_words=config.MAX_WORDS)
    tokenizer.fit_on_texts(train_data+test_data)
    train_tokenized = tokenizer.texts_to_sequences(train_data)
    test_tokenized = tokenizer.texts_to_sequences(test_data)
    train_data = sequence.pad_sequences(train_tokenized, maxlen = config.MAX_LENGTH)
    test_data = sequence.pad_sequences(test_tokenized, maxlen = config.MAX_LENGTH)
    return train_data, test_data, tokenizer.word_index

def get_word_embedding():
    data_path = 'data'
    # raw_embed = 'crawl-300d-2M.vec'
    raw_embed = 'glove.840B.300d.txt'
    EMBEDDING_FILE = os.path.join(data_path, raw_embed)
    embeddings_index = {}
    for line in open(EMBEDDING_FILE, "rb"):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    print (len(embeddings_index))
    return embeddings_index

def get_embed_matrix(embeddings_index, word_index):
    nb_words = min(config.MAX_WORDS, len(word_index))
    # embedding_matrix = np.zeros((nb_words, config.EMBEDDING_DIM))
    embedding_matrix = np.random.rand(nb_words, config.EMBEDDING_DIM)
    for word, i in word_index.items():
        if i >= config.MAX_WORDS:
            continue
        # embedding_vector = embeddings_index.get(str.encode(word))
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def fetch_data(aug=False):
    data_path = 'data'
    train = 'train.csv'
    test = 'test.csv'
    train_raw = get_raw_data(os.path.join(data_path, train))
    test_raw = get_raw_data(os.path.join(data_path, test))
    test_data = get_data(test_raw)

    if aug:
        train_de = 'train_de.csv'
        train_fr = 'train_fr.csv'
        train_es = 'train_es.csv'
        train_de_raw = get_raw_data(os.path.join(data_path, train_de))
        train_es_raw = get_raw_data(os.path.join(data_path, train_es))
        train_fr_raw = get_raw_data(os.path.join(data_path, train_fr))
        train_raw = pd.concat([train_raw, train_de_raw, train_es_raw, train_fr_raw]).drop_duplicates('comment_text')
    train_data = list(train_raw['comment_text'].fillna("_na_").values)
    train_label = get_label(train_raw)
        # print train_raw
        # train_de_data = get_data(train_de_raw)
        # train_de_label = get_label(train_de_raw)
        #train_es_data = get_data(train_es_raw)
        # train_es_label = get_label(train_es_raw)
        # train_fr_data = get_data(train_fr_raw)
        # train_fr_label = get_label(train_fr_raw)
        # train_data = train_data + train_de_data + train_fr_data + train_es_data
        # train_label = np.vstack((train_label, train_de_label, train_fr_label, train_es_label))

    train_data, test_data, word_index = process_data(train_data, test_data)
    return train_data, train_label, word_index

def fetch_test_data(aug=False):
    data_path = 'data'
    train = 'train.csv'
    test = 'test.csv'
    train_raw = get_raw_data(os.path.join(data_path, train))
    test_raw = get_raw_data(os.path.join(data_path, test))
    train_data = get_data(train_raw)
    test_data = get_data(test_raw)
    if aug:
        train_de = 'train_de.csv'
        train_fr = 'train_fr.csv'
        train_es = 'train_es.csv'
        train_de_raw = get_raw_data(os.path.join(data_path, train_de))
        train_es_raw = get_raw_data(os.path.join(data_path, train_es))
        train_fr_raw = get_raw_data(os.path.join(data_path, train_fr))
        train_raw = pd.concat([train_raw, train_de_raw, train_es_raw, train_fr_raw]).drop_duplicates('comment_text')
    train_data = list(train_raw['comment_text'].fillna("_na_").values)
    train_data, test_data, word_index = process_data(train_data, test_data)
    test_id = get_id(test_raw)
    return test_data, test_id

if __name__ == '__main__':
    # embedding_dict = get_word_embedding()
    # data, label, word_index = fetch_data()
    # print(np.sum(label, axis=0).astype(float) / label.shape[0])
    # em = get_embed_matrix(embedding_dict, word_index)
    # print(em.shape)
    # reverse_idx = {v:k for k,v in word_index.items()}
    # reverse_idx[0] = 'NOTHING'
    # for i in range(100):
    #     print [reverse_idx[v] for v in data[i] if v!=0]

    data_path = 'data'
    train = 'train.csv'
    test = 'test.csv'
    train_raw = pd.read_csv(os.path.join(data_path, train))
    raw_value = train_raw['comment_text'].fillna("_na_").values
    processed_data = []
    for i, v in enumerate(raw_value):
        text_parse(v)


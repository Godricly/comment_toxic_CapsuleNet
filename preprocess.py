import os
import numpy as np
import pandas as pd
from keras.preprocessing import text, sequence
import config

def get_raw_data():
    data_path = 'data'
    train = 'train.csv'
    test = 'test.csv'
    train_path = os.path.join(data_path, train)
    test_path = os.path.join(data_path, test)
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

def get_data(raw_data):
    return raw_data['comment_text'].fillna("_na_").values

def get_label(raw_data):
    labels = ['toxic', 'severe_toxic',
              'obscene', 'threat',
              'insult', 'identity_hate']
    return raw_data[labels].values

def process_data(train_data, test_data):
    tokenizer = text.Tokenizer(num_words=config.MAX_WORDS)
    tokenizer.fit_on_texts(list(train_data))
    train_tokenized = tokenizer.texts_to_sequences(train_data)
    test_tokenized = tokenizer.texts_to_sequences(test_data)
    train_data = sequence.pad_sequences(train_tokenized, maxlen = config.MAX_LENGTH)
    test_data = sequence.pad_sequences(test_tokenized, maxlen = config.MAX_LENGTH)
    return train_data, test_data, tokenizer.word_index

def get_word_embedding():
    data_path = 'data'
    EMBEDDING_FILE = os.path.join(data_path, 'glove.840B.300d.txt')
    embeddings_index = {}
    for line in open(EMBEDDING_FILE, "rb"):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    return embeddings_index

def get_embed_matrix(embeddings_index, word_index):
    nb_words = min(config.MAX_WORDS, len(word_index))
    embedding_matrix = np.zeros((nb_words, config.EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= config.MAX_WORDS:
            continue
        embedding_vector = embeddings_index.get(str.encode(word))
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def fetch_data():
    train_raw, test_raw = get_raw_data()
    train_data = get_data(train_raw)
    test_data = get_data(test_raw)
    train_label = get_label(train_raw)
    train_data, test_data, word_index = process_data(train_data, test_data)
    return train_data, train_label 

if __name__ == '__main__':
    embedding_dict = get_word_embedding()
    # em = get_embed_matrix(embedding_dict, word_index)
    # print(em.shape)

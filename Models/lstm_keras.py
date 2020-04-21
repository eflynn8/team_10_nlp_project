from __future__ import unicode_literals, print_function, division
from io import open
import numpy as np
import glob
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import utils
from random import seed
from random import sample
from pandas import DataFrame as df
import time
import string
import math
import matplotlib

def pick_samples():
    opts = set(np.arange(617))
    seed(1)
    picked = sample(opts, 432)
    picked = [('m' + str(i)) for i in picked]
    return picked

def make_label_tensor():
    samples = pick_samples()
    # print(samples)
    label_tensor = np.array([])
    metadata = open(curr_path + "/movie_titles_metadata.txt", "r", encoding="ISO-8859-1")
    
    genres = ['action', 'adult', 'adventure', 'animation', 'biography', 'comedy', 'crime','documentary', 'drama', 'family', 'fantasy', 'filmnoir', 'history', 'horror', 'music', 'musical', 'mystery', 'romance', 'scifi', 'short', 'sport', 'thriller', 'war', 'western']
    genre_idx = {}
    for i, gen in enumerate(genres):
        genre_idx[gen] = i

    for lines in metadata:
        fields = lines.split(" +++$+++ ")
        if fields[0] in samples:
            gen_vec = np.zeros(len(genres), dtype = int)
            labels = fields[5].translate(str.maketrans('', '', string.punctuation))
            labels = labels.rstrip('\n')
            labels = labels.split(' ')
            for l in labels:
                if l != '':
                    gen_vec[genre_idx[l]] = 1
            # print(gen_vec)
            label_tensor = np.append(label_tensor, gen_vec)
    label_tensor = label_tensor.reshape(len(samples), len(genres))

    return label_tensor

def make_dataframe():
    id_lines = {}
    text_file = open(curr_path + "/practice_scripts.txt", "r", encoding="ISO-8859-1")
    for line in text_file:
        fields = line.split(" +++$+++ ")
        if fields[2] not in id_lines:
            id_lines[fields[2]] = ['']
        id_lines[fields[2]][0] = id_lines[fields[2]][0] + fields[4]
    print(len(list(id_lines.values())))
    text_df = df(data = [list(id_lines.keys()), list(id_lines.values())], columns = ['ID', 'Lines'])
    return text_df

def file_to_features(path):
    features = utils.extract_features(path)
    return features

# def tokenize_corpus(text):
#     max_words = 500000
#     max_seq_len = 800
#     embedding_dim = 100

#     tokenizer = Tokenizer(num_words = max_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
#     tokenizer.fit_on_texts(text_df)
#     word_index = tokenizer.word_index
#     print('Found %s unique tokens' % len(word_index))
    

if __name__ == "__main__":
    start = time.time()
    os.chdir("..")
    curr_path = os.getcwd()
    os.chdir(curr_path + "/Corpus")
    curr_path = os.getcwd()

    output_tensor = make_label_tensor()
    # print("Label tensor", output_tensor)
    print("Label tensor shape", output_tensor.shape)

    max_words = 500000
    max_seq_len = 800
    embedding_dim = 100

    text_df = make_dataframe()
    print(text_df)
    # texts = np.array(list(text_df.values()), dtype = str)
    # print(type(texts))
    # print(texts.shape)
    str_texts = [texts[i][0] for i in range(len(texts))]

    tokenizer = Tokenizer(num_words = max_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(str_texts[0])
    word_index = tokenizer.word_index
    print('Found %s unique tokens' % len(word_index))

    X = tokenizer.texts_to_sequences(texts[0][0])

    for i in range(1, len(texts)):
        X = np.append(X, tokenizer.texts_to_sequences(texts[i][0]))
    # X.pad_sequences(X, max_len = max_seq_len)

    print('Shape of X: ', X.shape)



    features = file_to_features(curr_path + "/practice_scripts.txt")
    # print(features)

    input_tensor = np.array(list(features.values()), dtype = int)

    model = tf.keras.Sequential()
    model.add(layers.SpatialDropout1D(0.2))
    model.add(layers.LSTM(100, dropout = 0.2, recurrent_dropout = 0.2))
    model.add(layers.Dense(13, activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    epochs = 5
    batch_size = 20

    history = model.fit(input_tensor, output_tensor, epochs = epochs, batch_size = batch_size)
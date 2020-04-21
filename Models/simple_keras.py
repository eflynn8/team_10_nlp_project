from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, Embedding
from keras.optimizers import SGD
import os
import time
import numpy as np
import utils
from random import seed
from random import sample
from pandas import DataFrame as df
import time
import string
import math
import matplotlib

def create_model(input_dim, output_dim, lr, lr_decay, mom):
    model = Sequential()
    # model.add(Dense(500, activation = 'relu', input_dim = input_dim))
    # model.add(Dropout(0.1))
    model.add(Embedding(7, output_dim = 256))
    model.add(LSTM(128))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(Dense(output_dim, activation = 'sigmoid'))

    sgd = SGD(lr = lr, decay = lr_decay, momentum = mom)
    model.compile(loss = 'binary_crossentropy', optimizer = sgd)

    return model

def pick_samples():
    opts = set(np.arange(617))
    seed(1)
    picked = sample(opts, 10)
    picked = [('m' + str(i)) for i in picked]
    return picked

def make_label_tensor():
    samples = pick_samples()

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

start = time.time()

os.chdir("..")
curr_path = os.getcwd()
os.chdir(curr_path + "/Corpus")
curr_path = os.getcwd()

# features = utils.extract_features(curr_path + "/practice_scripts.txt")
# labels = make_label_tensor()

# X = np.array(list(features.values()), dtype = float)
# y = np.array(labels, dtype = float)

X = np.load(curr_path + "/train_features.npy")
print(X.size)
y = np.load(curr_path + "/train_labels.npy")

lr = 0.0001
lr_decay = 1e-6
mom = 0.9
num_epochs = 50
batch_size = 20

model = create_model(X.shape[1], y.shape[1], lr, lr_decay, mom)
model.fit(X, y, epochs = num_epochs, batch_size = batch_size)

preds = model.predict(X)
preds[preds >= 0.5] = 1
preds[preds < 0.5] = 0
print("Prediction: ", preds[2])
print("Correct: ", y[2])
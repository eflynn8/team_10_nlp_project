from __future__ import unicode_literals, print_function, division
from io import open
import numpy as np
import glob
import os
import torch
import torch.nn as nn
import utils
from random import seed
from random import sample
import time
import string
import math
import matplotlib

def read_file(path):
    return glob.glob(path)

def read_lines(filename):
    f = open(filename, "r", encoding="ISO-8859-1")
    lines = f.read().strip().split('\n')
    return lines

def id_to_genres(filename):
    id_genres = {}
    f = open(filename, "r", encoding="ISO-8859-1")

    for line in f:
        fields = line.split(" +++$+++ ")
        id_genres[fields[0]] = fields[5]
    return id_genres

def file_to_features(path):
    features = utils.extract_features(path)
    return features

def pick_samples():
    opts = set(np.arange(617))
    seed(1)
    picked = sample(opts, 10)
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

class Base_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Base_RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
n_features = 7
n_genres = 24
batch_size = 4
lr = 0.0005
n_iters = 100
print_every = 20
plot_every = 10

rnn = Base_RNN(n_features, n_hidden, n_genres)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr = lr)

# def train(genre_tensor, script_tensor):
#         hidden = rnn.initHidden()

#         rnn.zero_grad()

#         for i in range(script_tensor.shape[0]):
#             output, hidden = rnn(script_tensor[i], hidden)

#         loss = criterion(output, genre_tensor)
#         loss.backward()

#         for p in rnn.parameters():
#             p.data.add(-lr, p.grad.data)

#         return output, loss.item()

# n_hidden = 128
# rnn = Base_RNN(n_letters, n_hidden, n_categories)

if __name__ == "__main__":
    start = time.time()
    os.chdir("..")
    curr_path = os.getcwd()
    os.chdir(curr_path + "/Corpus")
    curr_path = os.getcwd()

    output_tensor = make_label_tensor()
    output_tensor = torch.from_numpy(output_tensor).float()
    # print("Label tensor", output_tensor)
    print("Label tensor shape", output_tensor.shape)

    features = file_to_features(curr_path + "/practice_scripts.txt")
    # print(features)

    input_tensor = np.array(list(features.values()), dtype = int)
    input_tensor = torch.from_numpy(input_tensor).float()
    # print("Input tensor: ", input_tensor)
    print("Input tensor shape", input_tensor.shape)
    end = time.time()
    print("Time to find features", end - start)

    ## Training the model
    train_start = time.time()

    for iter in range(n_iters):
        optimizer.zero_grad()
        full_output = torch.zeros(output_tensor.shape[0], output_tensor.shape[1])
        hidden = rnn.initHidden()

        for i in range(input_tensor.shape[0]):
            X = input_tensor[i]
            X = X.unsqueeze(0)
            y = output_tensor[i]
            # y = y.unsqueeze(0)
            print("Shape of X: ", X.shape)

            output, hidden = rnn(X, hidden)
            full_output[i] = output
            print("Output shape: ", output.shape)
            print(output)


            loss = criterion(output, y)
            print("Loss:", loss)
        loss.backward()
        optimizer.step()
        
        curr_loss += loss

        if iter % print_every == 0:
            print("Loss at " + iter, loss)
        
        print("Output: ", output)
        print("Output shape: ", output.shape)
        print("Final loss: ", loss.item())
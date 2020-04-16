from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import torch
import torch.nn as nn


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

def tokenize_scripts(lines):


def make_tensor(lines):

class Base_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Base_RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, hidden_size)
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
rnn = Base_RNN(n_letters, n_hidden, n_categories)
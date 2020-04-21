import os
import numpy as np
import utils
from random import seed
from random import sample
import string
import pickle

def pick_samples():
    opts = set(np.arange(617))
    seed(1)
    picked = sample(opts, 432)
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
    print(genre_idx)

    # with open(curr_path + '/genre_idx.pickle', 'wb') as f:
    # 	pickle.dump(genre_idx, f, protocol = pickle.HIGHEST_PROTOCOL)

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

os.chdir("..")
curr_path = os.getcwd()
os.chdir(curr_path + "/Corpus")
curr_path = os.getcwd()

labels = make_label_tensor()
# y = np.array(labels, dtype = float)
# np.save(curr_path + "/train_labels.npy", y)

# features = utils.extract_features(curr_path + "/train_scripts.txt")
# X = np.array(list(features.values()), dtype = float)
# np.save(curr_path + "/train_features.npy", X)
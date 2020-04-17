import numpy as np
import os
import matplotlib.pyplot as plt
import string


os.chdir("..")
curr_path = os.getcwd()
os.chdir(curr_path + "/Corpus")
curr_path = os.getcwd()
f = open(curr_path + "/movie_titles_metadata.txt", "r", encoding="ISO-8859-1")

genres_freq = {}

for lines in f:
    fields = lines.split(" +++$+++ ")
    labels = fields[5].translate(str.maketrans('', '', string.punctuation))
    labels = labels.rstrip('\n')
    labels = labels.split(' ')
    for l in labels:
        if l not in genres_freq:
            genres_freq[l] = 1
        else:
            genres_freq[l] = genres_freq[l] + 1

fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax = fig.add_subplot(111)
genres = list(genres_freq.keys())
genres.sort()
print(genres)
freq = [genres_freq[i] for i in genres]
print(freq)
plt.xticks(rotation = 90)
# plt.xticks(np.arange(len(genres)), genres)
ax.bar(genres, freq)
plt.show()
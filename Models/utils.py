import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm

import math
from utils import *
import numpy as np
import ast
import sys
import os


def preprocess(sent):
	sent = nltk.word_tokenize(sent)
	sent = nltk.pos_tag(sent)
	return sent

def extract_features(data):
	feat_len = 7
	id_to_lines = {}
	id_to_characters = {}
	with open(data, encoding='utf-8', errors='ignore') as file:
		for line in file:
			line = line.strip().split('+++$+++')
			movie_id = line[2][2:].strip()
			id_to_lines[movie_id] = ""
			id_to_characters[movie_id] = set()
	file.close()
	num_movies = len(id_to_lines)
	features = np.zeros((num_movies, feat_len+1))
	i = 0
	prev = ""
	with open(data, encoding='utf-8', errors='ignore') as file:
		for line in file:
			line = line.strip().split('+++$+++')
			movie_id = line[2][2:].strip()
			id_to_lines[movie_id] = id_to_lines[movie_id] + str(line[4]) + " "
			id_to_characters[movie_id].add(line[3])
			if movie_id != prev:
				features[i][0] = movie_id
				i += 1
				prev = movie_id
	file.close()

	for row in features:
		pos = preprocess(id_to_lines[str(int(row[0]))])
		num_words = len(pos)
		c = Counter(x[1] for x in pos)
		# POS Features
		row[1] = (c.get('JJ', 0) + c.get('JJR', 0) + c.get('JJS', 0)) / (c.get('NN', 0) + c.get('NNS', 0) + c.get('NNP', 0) + c.get('NNPS', 0))
		row[2] =(c.get('PRP', 0) + c.get('PRP$', 0)) / num_words
		row[3] = (c.get('WDT', 0) + c.get('WP', 0) + c.get('WP$', 0) + c.get('WRB', 0)) / num_words
		row[4] = (c.get('UH', 0) / num_words)

		nlp = en_core_web_sm.load()
		script = nlp(id_to_lines[str(int(row[0]))])
		labels = [x.label_ for x in script.ents]
		entities = Counter(labels)
		# NER Features
		row[5] = entities.get('LOC', 0)
		row[6] = entities.get('ORG', 0)

		row[7] = len(id_to_characters[str(int(row[0]))])

	thresholds = np.zeros((feat_len, 5))
	mean = np.mean(features[:,1:], 0)
	std = np.std(features[:,1:], 0)
	for i in range (feat_len):
		thresholds[i] = [mean[i] - 2*std[i], mean[i] - std[i], mean[i] + std[i], mean[i] + 2*std[i], mean[i] + 2*std[i]]

	# discretize features
	for i in range(num_movies):
		for j in range(1, feat_len+1):
			if features[i][j] < thresholds[j-1][0]:
				features[i][j] = 0
			elif features[i][j] < thresholds[j-1][1]:
				features[i][j] = 1
			elif features[i][j] < thresholds[j-1][2]:
				features[i][j] = 2
			elif features[i][j] < thresholds[j-1][3]:
				features[i][j] = 3
			else:
				features[i][j] = 4

	return features


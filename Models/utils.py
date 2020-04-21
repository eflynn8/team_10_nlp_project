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
from math import sqrt
from utils import *

import numpy as np
import ast


def preprocess(sent):
		sent = nltk.word_tokenize(sent)
		sent = nltk.pos_tag(sent)
		return sent

def extract_features(data):
		# pass
		features = {}
		id_to_lines = {}
		id_to_characters = {}
		with open(data, encoding='utf-8', errors='ignore') as file:
			for line in file:
				line = line.strip().split("+++$+++")
				id_to_lines[line[2]] = ""
				id_to_characters[line[2]] = set()
			file.close()
		with open(data, encoding='utf-8', errors='ignore') as file:
			for line in file:
				line = line.strip().split("+++$+++")
				id_to_lines[line[2]] = id_to_lines[line[2]] + str(line[4]) + " "
				id_to_characters[line[2]].add(line[3])
			file.close()
		# print(id_to_lines)

		for key in id_to_lines.keys():
			pos = preprocess(id_to_lines[key])
			num_words = len(pos)
			c = Counter(x[1] for x in pos)
			features[key] = [0, 0, 0, 0, 0, 0, 0]
			features[key][0] = (c.get('JJ', 0) + c.get('JJR', 0) + c.get('JJS', 0)) / (c.get('NN', 0) + c.get('NNS', 0) + c.get('NNP', 0) + c.get('NNPS', 0))
			features[key][1] =(c.get('PRP', 0) + c.get('PRP$', 0)) / num_words
			features[key][2] = (c.get('WDT', 0) + c.get('WP', 0) + c.get('WP$', 0) + c.get('WRB', 0)) / num_words

			nlp = en_core_web_sm.load()
			script = nlp(id_to_lines[key])
			labels = [x.label_ for x in script.ents]
			entities = Counter(labels)
			features[key][3] = entities.get('LOC', 0)
			features[key][4] = entities.get('ORG', 0)

			features[key][5] = len(id_to_characters[key])
			features[key][6] = (c.get('UH', 0) / num_words)

		# print(features)

		# calc mean and std dev
		thresholds = [0, 0, 0, 0, 0, 0, 0]
		for i in range (7):
			mean = sum(features[key][i] for key in features.keys()) / len(features)
			std = abs(sum(features[key][i] - mean for key in features.keys()))
			std = sqrt(pow(std, 2) / len(features))
			thresholds[i] = [mean - 2*std, mean - std, mean + std, mean + 2*std, mean + 2*std]
		# discretize features
		for key in features.keys():
			for i in range(len(features[key])):
				if features[key][i] < thresholds[i][0]:
					# features[key][i] = 'VL'
					features[key][i] = 0.2
				elif features[key][i] < thresholds[i][1]:
					# features[key][i] = 'L'
					features[key][i] = 0.4
				elif features[key][i] < thresholds[i][2]:
					# features[key][i] = 'AVG'
					features[key][i] = 0.6
				elif features[key][i] < thresholds[i][3]:
					# features[key][i] = 'H'
					features[key][i] = 0.8
				else:
					# features[key][i] = 'VH'
					features[key][i] = 1.0

		# print(features)
		return features
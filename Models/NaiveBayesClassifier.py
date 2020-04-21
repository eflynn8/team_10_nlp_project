import math
from utils import *
import numpy as np
import sys
import ast

class NaiveBayesClassifier():

	def __init__(self):
		self.logprior = {}
		self.loglikelihood = {}

	def fit(self, data, features, genres):
		np.set_printoptions(threshold=sys.maxsize)
		feat_len = len(features[0]) - 1
		genre_to_id = {g: [] for g in genres}
		num_movies = 0
		for row in data:
			num_movies += 1
			id_genres = ast.literal_eval(row[1])
			for g in id_genres:
				genre_to_id[g].append(row[0][1:])

		self.logprior = ({g: math.log(len(genre_to_id[g]) / num_movies) for g in genres} if len(genre_to_id[g]) > 0 else {g: 0})
		self.loglikelihood = {g: np.zeros((5, feat_len)) for g in genres}

		for genre in genre_to_id.keys():
			for movie_id in genre_to_id[genre]:
				for j in range(1, feat_len+1):
					if features[int(movie_id)][j] == 0:
						self.loglikelihood[genre][0][j-1] += 1
					elif features[int(movie_id)][j] == 1:
						self.loglikelihood[genre][1][j-1] += 1
					elif features[int(movie_id)][j] == 2:
						self.loglikelihood[genre][2][j-1] += 1
					elif features[int(movie_id)][j] == 3:
						self.loglikelihood[genre][3][j-1] += 1 
					else:
						self.loglikelihood[genre][4][j-1] += 1
			self.loglikelihood[genre] = np.add(self.loglikelihood[genre], 1.0)
			self.loglikelihood[genre] = np.log(np.divide(self.loglikelihood[genre], float(len(genre_to_id[genre]) + feat_len)))

	def predict(self, data, features):
		feat_len = len(features[0]) - 1
		pred_labels = [ [] for i in range(len(data)) ]
		correct_labels = [None] * len(data)
		genres = list(self.loglikelihood.keys())

		index = 0
		for row in data:
			movie_id = row[0][1:]
			correct_labels[index] = ast.literal_eval(row[1])
			f = features[int(movie_id)]
			f = f[1:]
			prob = np.zeros(len(genres))
			for i in range(len(genres)):
				s = 0.0
				for j in range(feat_len):
					s = s + self.loglikelihood[genres[i]][int(f[j])][j]
				prob[i] = self.logprior[genres[i]] + s

			Pthres = np.mean(prob)
			for i in range(len(prob)):
				if prob[i] >= Pthres:
					pred_labels[index].append(genres[i])
			index += 1

		return pred_labels, correct_labels, genres


from collections import Counter
import math
from utils import *
import numpy as np
import sys
import ast 
import time
import random
from NaiveBayesClassifier import NaiveBayesClassifier
import os


def accuracy(pred, labels, genres):
	correct = 0
	for i in range(len(pred)):
		correct += set(pred[i]) == set(labels[i])
	accuracy = correct/len(pred)
	print("Accuracy: %i / %i = %.4f " %(correct, len(pred), correct/len(pred)))

def fscore(pred, labels, genres):
	correct = Counter()
	guessed = Counter()
	guessed_correctly = Counter()
	for i in range(len(pred)):
		for j in range(len(pred[i])):
			guessed[pred[i][j]] += 1.0
		for k in range(len(labels[i])):
			correct[labels[i][k]] += 1.0
		common = list(set(labels[i]).intersection(pred[i]))
		for c in range(len(common)):
			guessed_correctly[common[c]] += 1.0

	genre_accuracy = {}

	true_pos = 0.0
	false_pos = 0.0
	false_neg = 0.0
	macro_precision = Counter()
	macro_recall = Counter()

	for g in genres:
		true_pos += guessed_correctly[g]
		false_pos += guessed[g] - guessed_correctly[g]
		false_neg += correct[g] - guessed_correctly[g]
		macro_precision[g] = (guessed_correctly[g] / guessed[g] if guessed[g] != 0 else 0.0)
		macro_recall[g] = (guessed_correctly[g] / correct[g] if correct[g] != 0 else 0.0)

		genre_accuracy[g] = (guessed_correctly[g] / guessed[g] if guessed[g] != 0 else 0.0)

	micro_precision = true_pos / (true_pos + false_pos)
	micro_recall = true_pos / (true_pos + false_neg)
	microf1score = (2.0 * micro_precision * micro_recall) / (micro_precision + micro_recall)
	macroprecision = sum(macro_precision.values()) / len(genres)
	macrorecall = sum(macro_recall.values()) / len(genres)
	macrof1score = (2.0 * macroprecision * macrorecall) / (macroprecision + macrorecall)

	print("Genre Accuracy")
	print(genre_accuracy)

	print("Micro-Averaged F1Score: %f" %(microf1score))
	print("Macro-Averaged F1Score: %f" %(macrof1score))


def main():

	np.set_printoptions(threshold=sys.maxsize)
	# separate data into training and testing
	training = np.zeros((1, 2), dtype=str)
	testing = np.zeros((1, 2), dtype=str)
	genres = set()

	os.chdir('../Corpus')
	with open('movie_titles_metadata.txt', encoding='utf-8', errors='ignore') as file:
		for line in file:
			line = line.strip().split(" +++$+++ ")
			for x in ast.literal_eval(line[5]):
				genres.add(x)
			arr = [[line[0], line[5]]]
			r = random.random()
			if r < 0.80:
				training = np.concatenate((training,arr),axis=0)
			else:
				testing = np.concatenate((testing,arr),axis=0)

	file.close()
	training = np.delete(training, 0, axis=0)
	testing = np.delete(testing, 0, axis=0)

	# train and test model
	start_time = time.time()
	features = extract_features('movie_lines.txt')
	model = NaiveBayesClassifier()
	print("TRAINING:")
	model.fit(training, features, genres)
	pred_labels, correct_labels, genres = model.predict(training, features)
	accuracy(pred_labels, correct_labels, genres)
	fscore(pred_labels, correct_labels, genres)
	print("TESTING:")
	pred_labels, correct_labels, genres = model.predict(testing, features)
	accuracy(pred_labels, correct_labels, genres)
	fscore(pred_labels, correct_labels, genres)

	print("Time for training and test: %.2f seconds" % (time.time() - start_time))


if __name__ == '__main__':
	main()

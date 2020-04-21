import os
import utils

os.chdir("..")
curr_path = os.getcwd()
os.chdir(curr_path + "/Corpus")
curr_path = os.getcwd()

features = utils.extract_features(curr_path + "/movie_lines.txt")
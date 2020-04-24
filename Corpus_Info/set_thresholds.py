import os

os.chdir("..")
curr_path = os.getcwd()
os.chdir(curr_path + "/Corpus")
curr_path = os.getcwd()

features = file_to_features(curr_path + "/movie_lines.txt")
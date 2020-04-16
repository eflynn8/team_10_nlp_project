import os

curr_path = os.getcwd()
os.chdir(curr_path + "/cornell_movie_dialogs_corpus")
f = open("movie_lines.txt", "r", encoding="ISO-8859-1")
script = open("m208_script", "w")

for line in f:
	fields = line.split(" +++$+++ ")
	if fields[2] == "m208":
		script.write(line)
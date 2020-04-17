import os

os.chdir("..")
curr_path = os.getcwd()
os.chdir(curr_path + "/Corpus")
curr_path = os.getcwd()
f = open(curr_path + "/movie_lines.txt", "r", encoding="ISO-8859-1")
script = open("m0_m208_script.txt", "w")

for line in f:
    fields = line.split(" +++$+++ ")
    if fields[2] == "m20":
        script.write(line)
    if fields[2] == "m208":
        script.write(line)
import os

curr_path = os.getcwd()
os.chdir(curr_path + "/cornell_movie_dialogs_corpus")
f = open("movie_titles_metadata.txt", "r", encoding="ISO-8859-1")
id_key = open("id_title.txt", "w")
title_key = open("title_genre.txt", "w") 

genres_dict = {}
id_title = {}
title_genre = {}

max_genres = 0
avg_genres = 0
i = 0

for x in f:
    fields = x.split(" +++$+++ ")
    id_title[fields[0]] = fields[1]
    title_genre[fields[1]] = fields[5]
    # print(fields)
    genres = fields[5].split(" ")
    if len(genres) > max_genres:
        max_genres = len(genres)
        print(genres)
        print(fields[1])
    avg_genres += len(genres)
    i += 1
print(max_genres)
print(avg_genres / i)

print(id_title, file = id_key)
print(title_genre, file = title_key)
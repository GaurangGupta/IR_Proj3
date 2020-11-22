import json
import numpy as np 

"""
	DESCRIPTION:
		Creates user-movie and movie-user matrix(stored as adjacency list) for both train
		and test data 

	INPUT:
		ratings.dat

	OUTPUT:
		train.dat
		test.dat


"""


train_test = ["train","test","ratings"]

#Construct user-movie and movie-user for both train and test
for f in train_test:
	#Open train/test file 
	file = open(f+".dat","r")
	if f=="ratings":
		name_file = ""
	else:
		name_file = "_" + f
	#Empty dictionary to store user-movie ratings
	user_movie_rating = {}

	for line in file.readlines():
		line = line.split()
		
		userId = line[0]
		movieId = line[1]
		rating = line[2]

		#Add movie to current user if already present
		if userId in user_movie_rating.keys():
			user_movie_rating[userId][movieId] = rating
		#Initialize user with current movie
		else:
			user_movie_rating[userId] = {}
			user_movie_rating[userId][movieId] = rating

	file.close()


	#Dumping the resultant dictionary into a json file
	with open('user_movie_rating'+name_file+'.json', 'w') as write_file:
		json.dump(user_movie_rating, write_file)


	#Open train/test file 
	file = open(f+".dat","r")
	
	#Empty dictionary to store movie-user ratings
	movie_user_rating = {}

	for line in file.readlines():
		line = line.split()
		
		userId = line[0]
		movieId = line[1]
		rating = line[2]

		#Add user to current movie if already present
		if movieId in movie_user_rating.keys():
			movie_user_rating[movieId][userId] = rating
		#Initialize movie with current user
		else:
			movie_user_rating[movieId] = {}
			movie_user_rating[movieId][userId] = rating

	file.close()


	#Dumping the resultant dictionary into a json file
	with open('movie_user_rating' +name_file+ '.json', 'w') as write_file:
		json.dump(movie_user_rating, write_file)

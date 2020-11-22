import json

"""
	DESCRIPTION:
		Calculates jaccard similarity between each pair of users

	
	Methods:
		Generates jaccard similarity between input users

	INPUT:
		user_movie_rating.json

	OUTPUT:
		jaccard_sim.json


"""

#Load the required file
user_movie_rating = json.load(open("user_movie_rating.json",'r'))


#Calculates jaccard similarity of usr1 and usr2
def jacc(usr1, usr2):
	lis1 = user_movie_rating[usr1].keys()
	lis2 = user_movie_rating[usr2].keys()
	num = len(set(lis1).intersection(lis2))
	den = len(list(set(lis1) | set(lis2)))

	return num/den

ct = 0
jaccard_sim= {}

#Computing jacc sim for users who have rated at least one same movie
for user1 in user_movie_rating.keys():
	print(ct)
	jaccard_sim[user1] = {}
	for user2 in user_movie_rating.keys():
		if user2 in jaccard_sim.keys():
			if user1 in jaccard_sim[user2].keys():
				jaccard_sim[user1][user2] = jaccard_sim[user2][user1]

		jaccard_sim[user1][user2] = jacc(user1,user2)

	ct += 1

#Dumping the resultant dictionary into a json file
with open('jaccard_sim.json', 'w') as write_file:
	json.dump(jaccard_sim, write_file)
import json 
import matplotlib.pyplot as plt
import time
import random

"""
	DESCRIPTION:
		Implements the collaborative filtering (and baseline) technique for recommender systems


	METHODS:
		precision_on_top_k(userid, k):
			Computes the precision on top k movies for the given userid

	INPUT:
		test.dat
		movie_user_rating_test.json
		movie_user_rating_train.json
		user_movie_rating_test.json
		user_movie_rating_train.json
		user_movie_rating.json
		movie_user_rating.json
		jaccard_sim.json

	OUTPUT:
	For both baseline and without baseline:
		RMSE 
		Precision on top K 
		Spearman Correlation 
		Time taken 

"""

st = time.time()

#Load the necessary files
test = open("test.dat","r")

movie_user_rating_test = json.load(open("movie_user_rating_test.json","r"))
movie_user_rating_train = json.load(open("movie_user_rating_train.json","r"))

user_movie_rating_test = json.load(open("user_movie_rating_test.json","r"))
user_movie_rating_train = json.load(open("user_movie_rating_train.json","r"))

user_movie_rating = json.load(open("user_movie_rating.json","r"))
movie_user_rating = json.load(open("movie_user_rating.json","r"))

jaccard = json.load(open('jaccard_sim.json','r'))


#Compute the global average
mu = 0
ct = 0
for user in user_movie_rating_train.keys():
	for movie in user_movie_rating_train[user].keys():
		mu += float(user_movie_rating_train[user][movie])
		ct += 1

#Global average
mu /= ct


#Compute user deviation from global mean rating
bx = {}
for user in user_movie_rating.keys():
	bx[user] = 0

	#Find average rating that the user has given to a movie
	for movie in user_movie_rating[user].keys():
		bx[user] += float(user_movie_rating[user][movie])

	bx[user] /= len(user_movie_rating[user])

	#Subtract global mean to find deviation of user
	bx[user] -= mu
 

#Compute movie deviation from global mean rating
bi = {}

#For each movie
for movie in movie_user_rating.keys():
	bi[movie] = 0

	#Find average rating that has been given to this movie
	for user in movie_user_rating[movie].keys():
		bi[movie] += float(movie_user_rating[movie][user])

	bi[movie] /= len(movie_user_rating[movie])

	#Subtract global mean to find deviation of movie 
	bi[movie] -= mu



def predict_rating(userid, movieId):
	"""
	Returns:
		The predicted value of rating 

	Input:
		userid : User id of the user whose rating has to be calculated
		movieId : movieid of the movie whose rating has to be calculated

	Output:
		rxi
		rxi_baseline
	"""
	jaccard_sims = {}
	rxi = 0
	rxi_baseline = 0
	
	#Load jaccard_sim.json for the train data values into a dictionary
	try:
		for key in movie_user_rating_train[movieId].keys():
			jaccard_sims[key] = jaccard[userid][key]
	except:
		pass
	

	num = 0
	den = 0
	#Finding weighted average (based on jacc similarity) of the rating
	for key in jaccard_sims.keys():
		num += float(user_movie_rating_train[key][movieId])*float(jaccard_sims[key])
		den += float(jaccard_sims[key])


	try:
		rxi = num/den
	except:
		rxi = 0


	num = 0
	den = 0


	for key in jaccard_sims.keys():
		bxj = bx[key] + mu + bi[movieId]
		num += float(float(user_movie_rating_train[key][movieId]) - float(bxj))*float(jaccard_sims[key])
		den += float(jaccard_sims[key])

		
	#Predicting rating using the baseline approach formula : bx = bxi + mu + bix
	bxi = bx[userid] + mu + bi[movieId]

	try:
		rxi_baseline = bxi + num/den
	except:
		rxi_baseline = bx[userid]

		
	return rxi, rxi_baseline



#To store MSE values
MSE = 0
MSE_baseline = 0
ind = 0

for cell in test.readlines():
	ind += 1
	cell = cell.split()

	cur_user = cell[0]	#current user_id
	movieId = cell[1]	#current movie_id

	#Predict rating for the current test cell (with and without baseline)
	rxi, rxi_baseline = predict_rating(cur_user,movieId)

	MSE += (rxi - float(cell[2]))**2
	MSE_baseline += (rxi_baseline - float(cell[2]))**2





def precision_on_top_k(userid, k):
	"""
	Returns:
		Precision on top k movies for the current user with and without baseline approach

	Input:
		userid : User id of the user whose precision has to be calculated
		k : Number of movies considered to find precision

	Output:
		Precision on top k
		Precision on top k for baseline
	"""
	
	#To store the movie ratings predicted for the current user (with and without baseline) 
	movie_sim_pred = {}
	movie_sim_pred_baseline = {}
	movie_sim_rat  = {}
	for movieId in user_movie_rating[userid].keys():
		jaccard_sims = {}

		try:
			for key in movie_user_rating[movieId].keys():
				jaccard_sims[key] = jaccard[userid][key]
		except:
			pass
		

		num = 0
		den = 0
		for key in jaccard_sims.keys():
			num += float(user_movie_rating[key][movieId])*float(jaccard_sims[key])
			den += float(jaccard_sims[key])

		try:
			rxi = num/den
		except:
			rxi = 0


		num = 0
		den = 0


		for key in jaccard_sims.keys():
			bxj = bx[key] + mu + bi[movieId]
			num += float(float(user_movie_rating[key][movieId]) - float(bxj))*float(jaccard_sims[key])
			den += float(jaccard_sims[key])

			
		bxi = bx[userid] + mu + bi[movieId]


		try:
			rxi_baseline = bxi + num/den
		except:
			rxi_baseline = bx[userid]

		movie_sim_pred[movieId] = rxi
		movie_sim_pred_baseline[movieId] = rxi_baseline
		movie_sim_rat[movieId]  = float(user_movie_rating[userid][movieId])

	movie_sim_pred = {k: v for k, v in sorted(movie_sim_pred.items(), key=lambda item: item[1])}
	movie_sim_pred_baseline = {k: v for k, v in sorted(movie_sim_pred_baseline.items(), key=lambda item: item[1])}
	movie_sim_rat = {k: v for k, v in sorted(movie_sim_rat.items(), key=lambda item: item[1])}


	based_on_pred = set()
	based_on_pred_baseline = set()
	based_on_rating  = set()

	for movieId in movie_sim_pred.keys():
		based_on_pred.add(movieId)
		if len(based_on_pred)==k:
			break

	for movieId in movie_sim_pred_baseline.keys():
		based_on_pred_baseline.add(movieId)
		if len(based_on_pred_baseline)==k:
			break

	for movieId in movie_sim_rat.keys():
		based_on_rating.add(movieId)
		if len(based_on_rating)==k:
			break

	return len(based_on_rating&based_on_pred)/len(based_on_rating) , len(based_on_rating&based_on_pred_baseline)/len(based_on_rating)


def Rand(start, end, num): 
	"""
	Returns:
		A list of num random numbers in the range (start, end)
	"""
    res = [] 
  
    for j in range(num): 
        res.append(random.randint(start, end)) 
  
    return res 






def calc_spearman(user1, user2):
	"""
	Returns:
		Spearman correlation of user1 and user2 for baseline and without baseline CF

	Input:
		user1 = userid of 1st user
		user2 = userid of 2nd user
	"""


	#Calculate ranks of movies for each user by storing them into dictionaries
	#and sorting the dictionaries by rating
	movie_sim_pred_u1 = {}
	movie_sim_pred_baseline_u1 = {}
	
	userid = str(user1)
	for movieId in user_movie_rating[str(user1)].keys():
		rxi, rxi_baseline = predict_rating(userid,movieId)
		movie_sim_pred_u1[movieId] = rxi
		movie_sim_pred_baseline_u1[movieId] = rxi_baseline


	movie_sim_pred_u1 = {k: v for k, v in sorted(movie_sim_pred_u1.items(), key=lambda item: item[1])}
	movie_sim_pred_baseline_u1 = {k: v for k, v in sorted(movie_sim_pred_baseline_u1.items(), key=lambda item: item[1])}



	movie_sim_pred_u2 = {}
	movie_sim_pred_baseline_u2 = {}

	
	userid = str(user2)
	for movieId in user_movie_rating[str(user2)].keys():
		rxi, rxi_baseline = predict_rating(userid,movieId)
		movie_sim_pred_u2[movieId] = rxi
		movie_sim_pred_baseline_u2[movieId] = rxi_baseline


	movie_sim_pred_u2 = {k: v for k, v in sorted(movie_sim_pred_u2.items(), key=lambda item: item[1])}
	movie_sim_pred_baseline_u2 = {k: v for k, v in sorted(movie_sim_pred_baseline_u2.items(), key=lambda item: item[1])}


	common_movies = list(set(movie_sim_pred_u1.keys()) & set(movie_sim_pred_u2.keys()))


	#Assigning ranks to each movie for each user (in both normal and baseline approach)
	ranks_u1 = {}
	ct = 0
	for movie in movie_sim_pred_u1.keys():
		if movie in common_movies:
			ranks_u1[movie] = ct
			ct += 1

	ranks_u1_baseline = {}
	ct = 0
	for movie in movie_sim_pred_baseline_u1.keys():
		if movie in common_movies:
			ranks_u1_baseline[movie] = ct
			ct += 1
	
	ranks_u2 = {}
	ct = 0
	for movie in movie_sim_pred_u2.keys():
		if movie in common_movies:
			ranks_u2[movie] = ct
			ct += 1

	ranks_u2_baseline = {}
	ct = 0
	for movie in movie_sim_pred_baseline_u2.keys():
		if movie in common_movies:
			ranks_u2_baseline[movie] = ct
			ct += 1


	MSE = 0
	for movie in common_movies:
		MSE += (ranks_u1[movie] - ranks_u2[movie])**2

	MSE_baseline = 0
	for movie in common_movies:
		MSE_baseline += (ranks_u1_baseline[movie] - ranks_u2_baseline[movie])**2

	n = len(common_movies)


	try:
		#Spearman formula
		spearman = 1 - (6*MSE)/((n)*(n*n-1))
		spearman_baseline = 1 - (6*MSE_baseline)/((n)*(n*n-1))
		return spearman, spearman_baseline
	except:
		return -1,-1



#Taking max precision on top 15 movies for 50 randomly generated users
sum_prec = 0
sum_baseline = 0
max_prec = 0
max_prec_baseline = 0
ct =0
indices = Rand(1,len(user_movie_rating),50)
for user in indices:
	user = str(user)
	ct += 1
	temp1, temp2 = precision_on_top_k(user,15)
	sum_prec += temp1
	sum_baseline += temp2
	max_prec = max(max_prec,temp1)
	max_prec_baseline = max(max_prec_baseline,temp2)



#Calculate spearman correlation for 50 randomly generated pair of users
i=0
avg_spearman = 0
avg_spearman_baseline = 0
while i<50:
	user1 = random.randint(1,2000)
	user2 = random.randint(1,2000)

	if user1!=user2:
		t1,t2 = calc_spearman(user1,user2)
		if t1!=-1:
			avg_spearman += t1
			avg_spearman_baseline += t2
			i += 1


print("RMSE = ",(MSE/ind)**(0.5))
print("RMSE_baseline = ",(MSE_baseline/ind)**(0.5))
print("Precision on top k = ",max_prec)
print("Precision on top k (Baseline) = ",max_prec_baseline)
print("Spearman",avg_spearman/50)
print("Spearman baseline",avg_spearman_baseline/50)
print("Time taken = ",time.time() -st)
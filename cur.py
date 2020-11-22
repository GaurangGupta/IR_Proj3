import numpy as np
import json 
import random
import time

"""
	DESCRIPTION:
		Implements CUR (+ 90% energy) technique for recommender systems


	METHODS:
		RMSE():
			Returns the RMSE values, given two user-movie matrices

		CUR():
			Returns C, W and R matrices for CUR decomposition
	
		form_user_matrix_train():
			Returns the 2d user-movie train matrix by reading the user_movie_rating_train.json file

		form_user_matrix_test():
			Returns the 2d user-movie test matrix by reading the user_movie_rating_test.json file
	
		SVD():
			Returns U, Sigma and Vt for the given input matrix

		retained_90():
			Returns modified U, Sigma and Vt by retaining 90% of the total energy

		precision_on_top_k(userid, k, A, reconstructed_A):
			Computes the precision on top k movies for the given userid

	INPUT:
		test.dat
		user_movie_rating_test.json
		user_movie_rating_train.json
		user_movie_rating.json

	OUTPUT:
	For SVD (+ 90 %):
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



def RMSE(A, reconstructed_A, ssq=0):
	"""
	Returns:
		RMSE of A and reconstructed_A

	set ssq = 1 to return MSE of A and reconstructed_A
	"""
	nofcells = A.shape[0]*A.shape[1]
	MSE = np.sum(np.square(A - reconstructed_A))/nofcells
	if ssq:
		return np.sum(np.square(A-reconstructed_A))
	return MSE**0.5


def SVD(A):
	"""
	Returns:
		U, Sigma and Vt after removing 0 values from Sigma

	Input:
		A : user-movie matrix
	"""

	#Computing eigen values to calculate U and S
	S,U = np.linalg.eig(np.dot(A,A.T))


	#Removing very small values from sigma
	for i in range(len(S)):
		if S[i]<1e-15:
			S[i] = 0
	S = np.sqrt(S)	

	#Sort sigma in descending order
	sorted_indices = S.argsort()[::-1]
	S = S[sorted_indices]
	U = U[:,sorted_indices]
	
	#Computing eigen values to calculate V
	_,V = np.linalg.eig(np.dot(A.T,A))
	V = V[:,sorted_indices]
	V = V.T

	sumsq = np.sum(np.square(S))

	cursumsq = 0
	for ind in range(len(S)):
		if cursumsq >= 0.9*sumsq:
			U = U[:,:ind]
			V = V[:ind,:]
			S = S[:ind]
			break

		cursumsq += S[ind]**2

	return np.real(U),np.real(S),np.real(V)



def CUR(A, c): 
	"""
	Returns:
		Returns C, W and R matrices of A for c rows and columns

	Input:
		A : user-movie matrix
		c : number of factors (rows and columns in R and C)

	Output:
		C, R, W
	"""

	#Calculating sum of squares along columns and rows of A
	A_sq = np.square(A)
	sum_A_sq = np.sum(A_sq)
	sum_A_sq_0 = np.sum(A_sq, axis=0)
	sum_A_sq_1 = np.sum(A_sq, axis=1)

	#Finding probability of each column and row
	P_x_c = sum_A_sq_0 / sum_A_sq
	P_x_r = sum_A_sq_1 / sum_A_sq

	#Initializing C, W and R as zero matrices
	row, col = A.shape
	C = np.zeros((row,c))
	R = np.zeros((c,col))
	W = np.zeros((c,c))

	# r_indices = []
	# c_indices = []
	#Choose columns randomly based on their probability to get selected 
	c_indices = np.random.choice(len(P_x_c),c,replace = False, p = P_x_c)
	r_indices = np.random.choice(len(P_x_r),c,replace = False, p = P_x_r)
	c_indices.sort()
	r_indices.sort()

	#Form C and R matrices 
	for i in range(c):
		C[:,i] = A[:, c_indices[i]] / ((c*P_x_c[c_indices[i]])**0.5)
		R[i,:] = A[r_indices[i],:] / ((c*P_x_r[r_indices[i]])**0.5)

	#Form W by intersection of the rows and columns selected, from A
	for i in range(c):
		for j in range(c):
			W[i][j] = A[r_indices[i]][c_indices[j]]

	return C, R, W


def retained_90(S,X,Yt):
	"""
	Returns:
		U, Sigma and Vt after removing 10% energy from sigma

	Input:
		X,S,Yt
	"""
	sumsq = np.sum(np.square(S))
	cursum = 0
	for ind in range(len(S)):
		if cursum >= 0.9*sumsq:
			X = X[:,:ind]
			Yt = Yt[:ind,:]
			S = S[:ind]
			return X,S,Yt

		cursum += S[ind]**2


def form_user_matrix_train():
	"""
	Returns:
		user-movie train matrix
	"""

	#Initialize empty 2d list
	A = [0]*(len(user_movie_rating))

	for i in range(0,len(A)):
		A[i] = [0]*(3952)

	#A[i][j] = rating if rated, else 0
	for user in user_movie_rating_train.keys():
		for movie in user_movie_rating_train[user].keys():
			A[int(user)-1][int(movie)-1] = np.double(user_movie_rating_train[user][movie])

	A = np.array(A)

	return A


def form_user_matrix_test():
	"""
	Returns:
		user-movie test matrix
	"""

	#Initialize empty 2d list
	A = [0]*(len(user_movie_rating))

	for i in range(0,len(A)):
		A[i] = [0]*(3952)

	#A[i][j] = rating if rated, else 0
	for user in user_movie_rating.keys():
		for movie in user_movie_rating[user].keys():
			A[int(user)-1][int(movie)-1] = float(user_movie_rating[user][movie])

	A = np.array(A)
	return A


def precision_on_top_k(userid, k, reconstructed_A, reconstructed_A_90):
	"""
	Returns:
		Precision on top k movies for the current user with and without 90% energy

	Input:
		userid : User id of the user whose precision has to be calculated
		k : Number of movies considered to find precision

	Output:
		Precision on top k
		Precision on top k for 90% energy
	"""

	#To store the movie ratings predicted for the current user (with and without 90% energy) 
	movie_sim_pred = {}
	movie_sim_pred_90 = {}
	movie_sim_rat  = {}
	for movieId in user_movie_rating[userid].keys():
		movie_sim_pred[movieId] = reconstructed_A[int(userid)-1][int(movieId)-1]
		movie_sim_pred_90[movieId] = reconstructed_A_90[int(userid)-1][int(movieId)-1]
		movie_sim_rat[movieId]  = float(user_movie_rating[userid][movieId])

	#Sort dictionaries based on ratings
	movie_sim_pred = {k: v for k, v in sorted(movie_sim_pred.items(), key=lambda item: item[1])}
	movie_sim_pred_90 = {k: v for k, v in sorted(movie_sim_pred_90.items(), key=lambda item: item[1])}
	movie_sim_rat = {k: v for k, v in sorted(movie_sim_rat.items(), key=lambda item: item[1])}

	#Take top k rated movies from the set of rated movies
	based_on_pred = set()
	based_on_pred_90 = set()
	based_on_rating  = set()

	for movieId in movie_sim_pred.keys():
		based_on_pred.add(movieId)
		if len(based_on_pred)==k:
			break

	for movieId in movie_sim_pred_90.keys():
		based_on_pred_90.add(movieId)
		if len(based_on_pred_90)==k:
			break

	for movieId in movie_sim_rat.keys():
		based_on_rating.add(movieId)
		if len(based_on_rating)==k:
			break

	#Calculating percentage of correctly suggested movies in top k
	return len(based_on_rating&based_on_pred)/len(based_on_rating) , len(based_on_rating&based_on_pred_90)/len(based_on_rating)



def calc_spearman(user1, user2, A_recon, A_recon_90):
	"""
	Returns:
		Spearman correlation of user1 and user2 for 90 and without 90 CF

	Input:
		user1 = userid of 1st user
		user2 = userid of 2nd user
	"""


	#Calculate ranks of movies for each user by storing them into dictionaries
	#and sorting the dictionaries by rating
	movie_sim_pred_u1 = {}
	movie_sim_pred_90_u1 = {}
	
	userid = str(user1)
	for movieId in user_movie_rating[str(user1)].keys():
		rxi = A_recon[int(userid)-1][int(movieId)-1]
		rxi_90 = A_recon_90[int(userid)-1][int(movieId)-1]

		movie_sim_pred_u1[movieId] = rxi
		movie_sim_pred_90_u1[movieId] = rxi_90


	movie_sim_pred_u1 = {k: v for k, v in sorted(movie_sim_pred_u1.items(), key=lambda item: item[1])}
	movie_sim_pred_90_u1 = {k: v for k, v in sorted(movie_sim_pred_90_u1.items(), key=lambda item: item[1])}



	movie_sim_pred_u2 = {}
	movie_sim_pred_90_u2 = {}

	
	userid = str(user2)
	for movieId in user_movie_rating[str(user2)].keys():
		rxi = A_recon[int(userid)-1][int(movieId)-1]
		rxi_90 = A_recon_90[int(userid)-1][int(movieId)-1]
		movie_sim_pred_u2[movieId] = rxi
		movie_sim_pred_90_u2[movieId] = rxi_90


	movie_sim_pred_u2 = {k: v for k, v in sorted(movie_sim_pred_u2.items(), key=lambda item: item[1])}
	movie_sim_pred_90_u2 = {k: v for k, v in sorted(movie_sim_pred_90_u2.items(), key=lambda item: item[1])}


	common_movies = list(set(movie_sim_pred_u1.keys()) & set(movie_sim_pred_u2.keys()))


	#Assigning ranks to each movie for each user (in both normal and 90 approach)
	ranks_u1 = {}
	ct = 0
	for movie in movie_sim_pred_u1.keys():
		if movie in common_movies:
			ranks_u1[movie] = ct
			ct += 1

	ranks_u1_90 = {}
	ct = 0
	for movie in movie_sim_pred_90_u1.keys():
		if movie in common_movies:
			ranks_u1_90[movie] = ct
			ct += 1
	
	ranks_u2 = {}
	ct = 0
	for movie in movie_sim_pred_u2.keys():
		if movie in common_movies:
			ranks_u2[movie] = ct
			ct += 1

	ranks_u2_90 = {}
	ct = 0
	for movie in movie_sim_pred_90_u2.keys():
		if movie in common_movies:
			ranks_u2_90[movie] = ct
			ct += 1


	MSE = 0
	for movie in common_movies:
		MSE += (ranks_u1[movie] - ranks_u2[movie])**2

	MSE_90 = 0
	for movie in common_movies:
		MSE_90 += (ranks_u1_90[movie] - ranks_u2_90[movie])**2

	n = len(common_movies)

	try:
		#Spearman formula
		spearman = 1 - (6*MSE)/((n)*(n*n-1))
		spearman_90 = 1 - (6*MSE_90)/((n)*(n*n-1))
		return spearman, spearman_90
	except:
		return -1,-1




def Rand(start, end, num): 
	"""
	Returns:
		A list of num random numbers in the range (start, end)
	"""
	res = [] 
  
	for j in range(num): 
		res.append(random.randint(start, end)) 
  
	return res 



#Form the train user-movie matrix
A_train = form_user_matrix_train()

#Perform CUR Decomposition
C,R, W  = CUR(A_train,900)

#Perform SVD on W matrix
X,S,Yt = SVD(W)

#Taking pseudo-inverse 
for i in range(len(S)):
	S[i] = 1/S[i]

S = np.diag(S,0)

#Calculating U matrix for CUR
U = np.dot(np.dot(Yt.T,S),X.T)

#Computing the reconstructed matrix from C, U and R
reconstructed_A = np.dot(np.dot(C,U),R)



#Retaining 90% energy in A and reconstructing again
X,S,Yt = retained_90(np.diag(S,0),X,Yt)
U = np.dot(np.dot(Yt.T,np.diag(S,0)),X.T)

#Computing the 90% energy reconstructed matrix
reconstructed_A_90 = np.dot(np.dot(C,U),R)



#Taking max precision on top 15 movies for 50 randomly generated users
max_prec = 0
max_prec_90 = 0
indices = Rand(1,len(user_movie_rating),50)
for user in indices:
	user = str(user)
	temp1, temp2 = precision_on_top_k(user,15 ,reconstructed_A,reconstructed_A_90)
	max_prec = max(max_prec,temp1)
	max_prec_90 = max(max_prec_90,temp2)



#Calculate spearman correlation for 50 randomly generated pair of users
i=0
avg_spearman = 0
avg_spearman_90 = 0
while i<50:
	user1 = random.randint(1,2000)
	user2 = random.randint(1,2000)

	if user1!=user2:
		t1,t2 = calc_spearman(user1,user2,reconstructed_A,reconstructed_A_90)
		if t1>0 and t2>0:
			avg_spearman += t1
			avg_spearman_90 += t2
			i += 1



A_test = form_user_matrix_test()
print("RMSE = ",RMSE(A_test,reconstructed_A))
print("RMSE 90 = ",RMSE(A_test,reconstructed_A_90))
print("Precision on top k = ",max_prec)
print("Precision on top k (90) = ",max_prec_90)
print("Spearman = ",avg_spearman/20)
print("Spearman 90 = ",avg_spearman_90/20)
print("\nTime taken =",time.time()-st)
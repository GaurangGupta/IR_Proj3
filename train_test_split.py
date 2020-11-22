import random 
  
"""
	DESCRIPTION:
		Creates 80-20 train-test split from the ratings file 

	
	Methods:
		Rand(start,end,num)
			Generates num random values between start and end

	INPUT:
		ratings.dat

	OUTPUT:
		train.dat
		test.dat


"""
 

#Open ratings file to count number of ratings
ratings = open('ratings.dat','r')
nolines = 0
for line in ratings.readlines():
	nolines += 1
ratings.close()



#Number of test ratings
test_count = nolines//5

#Randomly seleced test examples
indices = [i for i in range(nolines)]
random.shuffle(indices)

indices = indices[:test_count]
indices.sort()
#Open test.dat and train.dat for writing
test = open('test.dat','w')
train = open('train.dat','w')
ratings = open('ratings.dat','r')

#Convert indices to list


#Write to test.dat and train.dat
j=0
nolines = 0
for line in ratings.readlines():
	if nolines==indices[j]:
		test.write(line)
		if j<len(indices)-1:
			j += 1
	else:
		train.write(line)
	nolines += 1

#Close the files after writing
train.close()
test.close()
#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import loadmovielens as reader
import numpy as np
import math

ratings, movie_dictionary, user_ids, item_ids, movie_names = reader.read_movie_lens_data()


def Jaccard_Coefficient(movie_id_1, movie_id_2):
	movs1 = ratings[ratings[:,1]==movie_id_1]
	movs2 = ratings[ratings[:,1]==movie_id_2]
	m1m2 = np.intersect1d(movs1[:,0], movs2[:,0])
	return round((len(m1m2) / (len(movs1)+len(movs2)-len(m1m2) + 0.0)), 3)


def Correlation_Coefficient(movie_id_1, movie_id_2, ratecount):
	movies1 = ratings[ratings[:,1]==movie_id_1]
	movies2 = ratings[ratings[:,1]==movie_id_2]

	mov1 = movies1[np.in1d(movies1[:,0], movies2[:,0])]
	mov2 = movies2[np.in1d(movies2[:,0], movies1[:,0])]

	if len(mov1) < ratecount:
		return 0.0

	vals = np.zeros((2, len(mov1)))
 	vals[0] = mov1[np.argsort(mov1[:, 0])][:,2]
 	vals[1] = mov2[np.argsort(mov2[:, 0])][:,2]

	coeff = np.corrcoef(vals)[0][1]

	if math.isnan(coeff):
		return 0.0

	return round(coeff, 3)
	#return round(np.corrcoef(vals)[0][1], 3)




def top_five(mov_id, ratecount, func):
	coeffs = []
	top5movies = []
	
	for m in movie_dictionary:
		if mov_id != m:
			if(func == "corr"):
				val = Correlation_Coefficient(mov_id, m, ratecount)
			else: 
				val = Jaccard_Coefficient(mov_id, m)	

			if not math.isnan(val):
				coeffs.append((val, m))
			else:
				print m
	
	coeffs.sort(reverse=True)
	
	for i in range(7):
		top5movies.append(str(i+1)+". "+ movie_dictionary[coeffs[i][1]] +", "+str(coeffs[i][0])) 

	print_top5(top5movies, mov_id, ratecount)
	return top5movies 
	

def print_top5(topmovies, cur_id, ratecount):
	print "top 5 matches for ", movie_dictionary[cur_id], "using ratecount", ratecount
	for m in topmovies:
		print m

def countusercount(mov_id, func):
	top_five(mov_id, 5, func)
	top_five(mov_id, 10, func)
	top_five(mov_id, 20, func)
	

print "jacc: "
top_five(191, 4, "jacc")

print "corr: "
countusercount(191, "corr")


print "jacc: "
top_five(250, 4, "jacc")

print "corr: "
countusercount(250, "corr")



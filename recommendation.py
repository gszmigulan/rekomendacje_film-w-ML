import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def create_my_ratings(m,user_id):
    with open('ml-latest-small/ratings.csv', newline='') as f:
     reader= csv.reader(f)
     data = list(reader)
     lista= [[0.0]*m for i in range(1)]
     max_id=1
   
     for i in range(1,len(data)): 
        if int(data[i][0])==user_id:
          film_id = int(data[i][1])
          if 0 < film_id < m:
            lista[int(data[i][0])-1][film_id - 1]= float(data[i][2])
     l = np.array(lista).T
     
     return l

def create_table( m, my_ratings):
   
   with open('ml-latest-small/ratings.csv', newline='') as f:
     reader= csv.reader(f)
     data = list(reader)
     lista= [[0.0]*m for i in range(611)]
     max_id=1
   
     for i in range(1,len(data)): 
        film_id = int(data[i][1])
        if 0 < film_id < m:
          lista[int(data[i][0])-1][film_id - 1]= float(data[i][2])
          if film_id > max_id :
             max_id = film_id

   x = np.array(lista)
   y= np.array(my_ratings)
   np.seterr(divide="ignore", invalid="ignore")
   has= np.nan_to_num(np.array(y)/np.linalg.norm(y))
   z= np.dot( np.nan_to_num(x/np.linalg.norm(x,axis=0)), has)
   z_norm = np.nan_to_num(z / np.linalg.norm(z))
   X = np.nan_to_num(x/ np.linalg.norm(x, axis=0))
   Z = z_norm
   R = np.dot(X.T, Z) 

   film_id= [None]*m
   with open('ml-latest-small/movies.csv', newline='') as f:
     reader= csv.reader(f)
     data = list(reader)
     for i in range(1,len(data)):
        if(int(data[i][0])< 10000):
           film_id[int(data[i][0])-1]=data[i][1]

   result= [[0.0, None] for i in range(9019)]
   for i in range(1, 9019):
      if film_id[i-1] is not None:
        result[i-1][0]=float(R[i-1])
        result[i-1][1]= film_id[i-1]
   sorted_f=sorted(result, key = lambda x: x[0])
   

   for i in reversed(range(len(sorted_f))):
      if sorted_f[i][1] is not None:
        print(sorted_f[i])


films_nr= 9019
my_ratings=np.zeros((films_nr,1))
my_ratings[2571-1]=5
my_ratings[32-1]=4
my_ratings[260-1]=5
my_ratings[1097-1]=4


user_ratings = create_my_ratings(films_nr, 1) # recommendation for user nr 1
create_table(films_nr, user_ratings)








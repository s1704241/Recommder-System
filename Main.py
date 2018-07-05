# -*- coding: utf-8 -*-
print(__doc__)

from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from Autoencoder import Classifier
from tensorflow.examples.tutorials.mnist import input_data
from collections import Counter  
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score  

np.random.seed(7)

#Pre-processing
#Read the data
udata = pd.read_csv('sushi3.udata', sep='\t', encoding='utf8')
idata = pd.read_csv("sushi3.idata", sep='\t', encoding='utf8')
score = pd.read_csv('sushi3b.5000.10.score',  sep=' ', header=None, encoding='utf8')
score = score.values
#Pick out goal data and do normalization
ucolumn = ["gender","age","time","prefectureID","RegionID","directionID","currentPID","currentRID","currentDID","SameID"]
udata = scale(udata[ucolumn])
icolumn = ["style","major group","minor group","heaviness","consumption frequency","normalized price","sell frequency"]
idata = scale(idata[icolumn])
#
#x = udata
#clusters=[2,3,4,5,6,7,8,9,10]
#subplot_counter=1
#sc_scores=[]
#
#for t in clusters:
#    kmeans_model = KMeans(n_clusters=t).fit(x)
#    sc_score = silhouette_score(x, kmeans_model.labels_, metric='euclidean')
#    sc_scores.append(sc_score)
#plt.figure()
#
##Draw the gragh and show the silhouette score of each cluster.
#plt.plot(clusters,sc_scores,'*-')
#plt.xlabel('Numbers of clusters')
#plt.ylabel('Silhouette Coefficient score')
#plt.show()
#
##Do K-means on users
#userkmeans = KMeans(n_clusters=3, random_state=0).fit(udata)
#typecount = Counter(userkmeans.labels_)
#cluster = 3
##Key represents the usetype, value represents the score
#usertype = {}
##Key represents the usetype, value represents the item feature
#itemtype = {}
##Key represents the usetype, value represents the item feature with score
#item = {}
##The type of the users
#utype = userkmeans.labels_
##Split users into different groups
#for i in range(cluster):
#    usertype[i] = score[utype == i, :]

t = np.ones([100])
zero = t.reshape(-1,1)
test = np.concatenate([idata, zero],axis=1)

#Do Autoencoder and reduce dimensionality
DNN = Classifier(batch_size=20, learning_rate=0.01, training_epochs=20, dropout_rate=0.0,
                  batch_norm_use=False)

autoencoder = DNN(input_data= test , display_step = 1,n_input = 8)

#
##Add score into item features
#for j in range(cluster):
#    usertype[j] = usertype[j].reshape(-1,1)
#    itemtype[j] = np.repeat(idata, typecount[j], axis=0)
#    item[j] = np.concatenate([itemtype[j], usertype[j]],axis=1)
#    item[j] = item[j][item[j][:,7]!=-1,:]
#    
#    
##Do K-means on item features
#for n in range(len(item)):
#    y = item[0]
#    clusters=[2,3,4,5,6,7,8,9,10]
#    subplot_counter=1
#    sc_scores=[]
#    
#    for t in clusters:
#        kmeans_model = KMeans(n_clusters=t).fit(y)
#        sc_score = silhouette_score(y, kmeans_model.labels_, metric='euclidean')
#        sc_scores.append(sc_score)
#    plt.figure()
#    
#    #Draw graphs
#    plt.plot(clusters,sc_scores,'*-')
#    plt.xlabel('Numbers of clusters')
#    plt.ylabel('Silhouette Coefficient score')
#    plt.show()
#
#itemcount = {}
#itemkmeans = {}
#for m in range(3):
#    itemkmeans[m] = KMeans(n_clusters=4, random_state=0).fit(item[m])
#    itemcount[m] = Counter(itemkmeans[m].labels_)
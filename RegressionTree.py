# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 14:33:23 2018

@author: hasee
"""

print(__doc__)

# Import the necessary modules and libraries
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
import collections
import pandas as pd
import pydotplus
import random

from Online_updating import Belief_Updating
from IPython.display import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from collections import Counter  
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score 
from sklearn.tree import export_graphviz





data = "sushi"
#data = "movielens"
if data == "sushi":
    #Pre-processing
    #Read the data
    idata = pd.read_csv("sushi3.idata", sep='\t', encoding='utf8')
    score = pd.read_csv('sushi3b.5000.10.score',  sep=' ', header=None, encoding='utf8')
    score = score.values
    icolumn = ["style","major group","minor group","heaviness","consumption frequency","normalized price","sell frequency"]
    idata = scale(idata[icolumn])
    #PCA reduce dimension
    pca = PCA(n_components=50, svd_solver='full')
    #print(pca.explained_variance_ratio_)
    test = score[int(score.shape[0]*0.9):,:]


if data == "movielens":
#    #Pre-processing
#    #Read the data
    idata = pd.read_csv("genome-scores.csv", sep=',', encoding='utf8').values
    idata = idata[:1128*9000,2].reshape(-1,1128)
    score = pd.read_csv('Movie_score.csv',  encoding='utf8').values[:,1:]
    
    pca = PCA(n_components=1400, svd_solver='full')
    #print(pca.explained_variance_ratio_) 
    test = score[int(score.shape[0]*0.9):,:]
    
pca.fit(score)                 
pca_score = pca.transform(score) 
x = pca_score
clusters=[2,3,4,5,6,7,8,9,10]
subplot_counter=1
sc_scores=[]



for t in clusters:
    kmeans_model = KMeans(n_clusters=t).fit(x)
    sc_score = silhouette_score(x, kmeans_model.labels_, metric='euclidean')
    sc_scores.append(sc_score)
plt.figure()

#Draw the gragh and show the silhouette score of each cluster.
plt.plot(clusters,sc_scores,'*-')
plt.xlabel('Numbers of clusters')
plt.ylabel('Silhouette Coefficient score')
plt.show()

if data == "sushi":
    #Do K-means on users
    userkmeans = KMeans(n_clusters=8, random_state=0).fit(x)
    typecount = Counter(userkmeans.labels_)
    user_cluster = 8


if data == "movielens":
    #Do K-means on users
    userkmeans = KMeans(n_clusters=4, random_state=0).fit(x)
    typecount = Counter(userkmeans.labels_)
    user_cluster = 4
    
#Key represents the usetype, value represents the score
usertype = {}
#Key represents the usetype, value represents the item feature
itemtype = {}


#The type of the users
utype = userkmeans.labels_
#Split users into different groups
for i in range(user_cluster):
    usertype[i] = score[utype == i, :]
#Concate the item features and the ratings
for j in range(user_cluster):
    itemtype[j] = usertype[j].reshape(-1,1)
    itemtype[j] = np.concatenate([np.tile(idata,(usertype[j].shape[0],1)), itemtype[j]],axis=1)
    itemtype[j] = itemtype[j][itemtype[j][:,-1]!=-1,:]


#Do Regression tree
item = []
regression = []
for model in range(len(itemtype)):
    target = itemtype[model][:,-1]
    train_data = itemtype[model][0:int(itemtype[model].shape[0]*0.9),0:-1]
    train_label = target[0:int(itemtype[model].shape[0]*0.9)]
    X = train_data
    y = train_label
    
    # Fit regression model
    if data == "sushi":
        regression.append(tree.DecisionTreeRegressor(max_depth=4,min_samples_split=10))
        regression[model].fit(X, y)
        y = regression[model].predict(idata)
    
    
    if data == "movielens":
        regression.append(tree.DecisionTreeRegressor(max_depth=7,min_samples_split=100))
        regression[model].fit(X, y)
        y = regression[model].predict(idata)


    samples = collections.defaultdict(list)
    dec_paths = regression[model].decision_path(idata)
    
    for d, dec in enumerate(dec_paths):
        for i in range(regression[model].tree_.node_count):
            if dec.toarray()[0][i]  == 1:
                samples[i].append(d) 
    item.append(samples)

model = 0
dot_data = tree.export_graphviz(regression[model], out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png())

#graph.write_png("tree.png")	# 生成png文件
#graph.write_jpg("tree.jpg")	# 生成jpg文件
#graph.write_pdf("tree.pdf")	# 生成pdf文件

#obtain class labels
model = []
if data == "sushi":
    classes = {
            0:[3,6,11,12,13,18,21,26,27,28],
            1:[3,6,11,12,14,15,19,20,22,23,24],
            2:[3,7,8,9,14],
            3:[3,7,8,11,12,13,16],
            4:[3,7,8,9,15,16,18,19,20],
            5:[4,5,6,11,12,14,15,16],
            6:[3,6,11,12,14,15,17,24,25,26],
            7:[3,7,8,10,14,15,18,21,24]
        }
    for i in range(len(classes)):
        model.append([])
    
if data == "movielens":
    classes = {
            0:[3,6,11,12,13,18,21,26,27,28],
            1:[3,6,11,12,14,15,19,20,22,23,24],
            2:[3,7,8,9,14],
            3:[3,7,8,11,12,13,16],
            4:[3,7,8,9,15,16,18,19,20],
            5:[4,5,6,11,12,14,15,16],
            6:[3,6,11,12,14,15,17,24,25,26],
            7:[3,7,8,10,14,15,18,21,24]
        }
    for i in range(len(classes)):
        model.append([])
    
#Obtain the item class labels of every model
item_class = {}    
for m in range(len(classes)):
    item_class[m] = np.array([None,None]).reshape(1,-1)
    for key in range(len(classes[m])):
        location= classes[m][key]
        class_name = np.ones((len(item[m][location]),1))*key
                            
        label = np.concatenate((np.array(item[m][location]).reshape(-1,1),class_name),axis=1)
        item_class[m] = np.concatenate((item_class[m],label),axis=0)
        
        #Remove the None on the top
    item_class[m] = item_class[m][1:,:]
    item_class[m] = item_class[m][item_class[m][:,0].argsort()]
    item_class[m] = item_class[m][:,1]

#Calculate the prior distribution the score
means = []
if data == "sushi":
    for types in range(user_cluster):
        mask = (usertype[types]<0)
        means.append(np.ma.masked_where(mask, usertype[types]).mean(axis=0).data)

if data == "sushi":
    for types in range(user_cluster):
        mask = (usertype[types]<=0)
        means.append(np.ma.masked_where(mask, usertype[types]).mean(axis=0))

item_rating = {}
for i in range(user_cluster):
    model_means = np.array(means[i]).reshape(-1,1)
    item_rating[i] = (np.concatenate((idata,model_means),axis=1))
    for j in range(len(list(Counter(item_class[i])))):
        mean = np.mean(item_rating[i][item_class[i]==j,-1],axis=0)
        var = np.var(item_rating[i][item_class[i]==j,-1],axis=0)
        model[i].append([mean,var])
    

#ONline Procedure
#A list that store all models, in each model,each list contains 
#the mean and variance of gaussian distribution respectively
item_distribution = []
total_loss = 0
iteration = 0
if data == "sushi":
    for time_step in range(test.shape[0]):
        filters = test[time_step,:]>=0
        index = np.array(range(test.shape[1]))
        candidate = filters*index
        candidate = candidate[candidate>0]
        column = random.sample(list(candidate),1)
        userResponse = test[time_step,column]
        item_distribution = []
        for k in range(user_cluster):
            label = int(item_class[k][column][0])
            item_distribution.append(model[k][label])
        
        #Compare and pick out the most close model to the userResponse
        array = np.array(item_distribution)
        distance = abs(array - userResponse)
        loc = np.where(distance==np.min(distance))[0][0]
        mean = model[loc][int(item_class[loc][column])][0]
        variance = model[loc][int(item_class[loc][column])][1]
        
        #Updating the prior distribution
        Updating =  Belief_Updating(responseVariance=0.01)
        posterior = Updating(mean, variance, userResponse)
        posteriorMean = posterior[0]
        posteriorVar = posterior[1]
        model[loc][int(item_class[loc][column])] = posterior
    
        #Set recommendation online
        current_model = np.array(model[loc])
        current_model_mean = current_model[:,0]
        candidate_class = np.where(current_model==np.max(current_model))[0][0]
        mask = item_class[loc]==candidate_class
        position = np.array(range(len(item_class[loc])))
        new_candidate = mask*position
        new_candidate = new_candidate[new_candidate>0]
        intersection = list(set(candidate).intersection(set(new_candidate)))
        if len(intersection)>0:
            column = random.sample(intersection,1)
            iteration += 1
        
            #Calculate the loss of recommendation
            pred_rating = current_model_mean[candidate_class]
            true_rating = test[time_step,column]
            loss = abs(pred_rating - true_rating)
            total_loss = loss + total_loss
            averge_loss = total_loss/iteration
            print ("Avg Loss:",averge_loss)
    



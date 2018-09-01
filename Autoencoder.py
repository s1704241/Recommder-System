# -*- coding: utf-8 -*-
print(__doc__)


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from Online_updating import Belief_Updating
from Network import Autoencoder
from collections import Counter  
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score  

np.random.seed(7)

#Pre-processing
#Read the data

data = "sushi"
#data = "movielens"
if data == "sushi":
    udata = pd.read_csv('sushi3.udata', sep='\t', encoding='utf8')
    idata = pd.read_csv("sushi3.idata", sep='\t', encoding='utf8')
    score = pd.read_csv('sushi3b.5000.10.score',  sep=' ', header=None, encoding='utf8')
    score = score.values
    np.savetxt('Sushi_score.csv', score, delimiter = ',')
    #Pick out goal data and do normalization
    ucolumn = ["gender","age","time","prefectureID","RegionID","directionID","currentPID","currentRID","currentDID","SameID"]
    udata = scale(udata[ucolumn])
    icolumn = ["style","major group","minor group","heaviness","consumption frequency","normalized price","sell frequency"]
    idata = scale(idata[icolumn])
    test = score[int(score.shape[0]*0.9):,:]


x = udata
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

#Do K-means on users
userkmeans = KMeans(n_clusters=3, random_state=0).fit(udata)
typecount = Counter(userkmeans.labels_)
cluster = 3
#Key represents the usetype, value represents the score
usertype = {}
##Key represents the usetype, value represents the item feature
#itemtype = {}
#Stores different item features for inverse user type
item = []


#The type of the users
utype = userkmeans.labels_
#Split users into different groups
for i in range(cluster):
    usertype[i] = score[utype == i, :]

#Calculate the prior distribution the score
mean_1 = []
mean_2 = []
mean_3 = []

std_1 = []
std_2 = []
std_3 = []
for j in range(score.shape[1]):
    column_1 = usertype[0][:,j].reshape(1,-1)
    mean_1.append(column_1[:,column_1[0,:]!=-1].mean())
    std_1.append(column_1[:,column_1[0,:]!=-1].std())
    
    column_2 = usertype[0][:,j].reshape(1,-1)
    mean_2.append(column_2[:,column_2[0,:]!=-1].mean())
    std_2.append(column_2[:,column_2[0,:]!=-1].std())
    
    column_3 = usertype[0][:,j].reshape(1,-1)
    mean_3.append(column_3[:,column_3[0,:]!=-1].mean())
    std_3.append(column_3[:,column_3[0,:]!=-1].std())

mean_score_1 = np.array(list(mean_1)).reshape(-1,1)
mean_score_2 = np.array(list(mean_2)).reshape(-1,1)
mean_score_3 = np.array(list(mean_3)).reshape(-1,1)

training_1 = np.concatenate([idata, mean_score_1],axis=1)
training_2 = np.concatenate([idata, mean_score_2],axis=1)
training_3 = np.concatenate([idata, mean_score_3],axis=1)

#Do Autoencoder and reduce dimensionality for thre user types
DNN_1 = Autoencoder(batch_size=20, learning_rate=0.05, training_epochs=20, dropout_rate=0.0,
                  batch_norm_use=False)
autoencoder_1 = DNN_1(input_data= training_1 , display_step = 1,n_input = 8)
item.append(autoencoder_1)

DNN_2 = Autoencoder(batch_size=20, learning_rate=0.05, training_epochs=20, dropout_rate=0.0,
                  batch_norm_use=False)
autoencoder_2 = DNN_2(input_data= training_2 , display_step = 1,n_input = 8)
item.append(autoencoder_2)

DNN_3 = Autoencoder(batch_size=20, learning_rate=0.05, training_epochs=20, dropout_rate=0.0,
                  batch_norm_use=False)
autoencoder_3 = DNN_3(input_data= training_3 , display_step = 1,n_input = 8)
item.append(autoencoder_3)

    
    
#Do K-means on item features
for n in range(len(item)):
    y = item[n]
    clusters=[2,3,4,5,6,7,8,9,10]
    subplot_counter=1
    sc_scores=[]
    
    for t in clusters:
        kmeans_model = KMeans(n_clusters=t).fit(y)
        sc_score = silhouette_score(y, kmeans_model.labels_, metric='euclidean')
        sc_scores.append(sc_score)
    plt.figure()
    
    #Draw graphs
    plt.plot(clusters,sc_scores,'*-')
    plt.xlabel('Numbers of clusters')
    plt.ylabel('Silhouette Coefficient score')
    plt.show()

cluster = [7,6,4]
itemkmeans = {}
for m in range(3):
    itemkmeans[m] = KMeans(n_clusters=cluster[m], random_state=0).fit(item[m])
#    itemcount[m] = Counter(itemkmeans[m].labels_)    
    label_pred = itemkmeans[m].labels_ #获取聚类标签
    centroids = itemkmeans[m].cluster_centers_ #获取聚类中心
    inertia = itemkmeans[m].inertia_ # 获取聚类准则的总和
    mark = ['or', '^g', '+b', 'sk', 'dr', '<r', 'pr','ob', 'og', 'ok',]
    #这里'or'代表中的'o'代表画圈，'r'代表颜色为红色，后面的依次类推
    j = 0 
    for i in label_pred:
        plt.plot([item[m][j:j+1,0]], [item[m][j:j+1,1]], mark[i], markersize = 5)
        
        j +=1
    plt.savefig(str(m)+"space.pdf")
    plt.show()

output = {}
model = []
for i in range(len(cluster)):
    output[i] = itemkmeans[i].labels_.reshape(-1,1)
    output[i] = np.concatenate([np.array(range(0,100)).reshape(-1,1), output[i]],axis=1)
    np.savetxt('output'+str(i)+'.csv', output[i], delimiter = ',')
    model.append([])

user_cluster = 3
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
    for j in range(len(list(Counter(itemkmeans[i].labels_)))):
        mean = np.mean(item_rating[i][itemkmeans[i].labels_==j,-1],axis=0)
        var = np.var(item_rating[i][itemkmeans[i].labels_==j,-1],axis=0)
        model[i].append([mean,var])
    

#ONline Procedure
#A list that store all models, in each model,each list contains 
#the mean and variance of gaussian distribution respectively

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
            label = itemkmeans[k].labels_[column][0]
            item_distribution.append(model[k][label])
        
        #Compare and pick out the most close model to the userResponse
        array = np.array(item_distribution)
        distance = abs(array - userResponse)
        loc = np.where(distance==np.min(distance))[0][0]
        mean = model[loc][int(itemkmeans[loc].labels_[column])][0]
        variance = model[loc][int(itemkmeans[loc].labels_[column])][1]
        
        #Updating the prior distribution
        Updating =  Belief_Updating(responseVariance=0.01)
        posterior = Updating(mean, variance, userResponse)
        posteriorMean = posterior[0]
        posteriorVar = posterior[1]
        model[loc][int(itemkmeans[loc].labels_[column])] = posterior
    
        #Set recommendation online
        current_model = np.array(model[loc])
        current_model_mean = current_model[:,0]
        candidate_class = np.where(current_model==np.max(current_model))[0][0]
        mask = itemkmeans[loc].labels_==candidate_class
        position = np.array(range(len(itemkmeans[loc].labels_)))
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
    

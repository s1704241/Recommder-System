# -*- coding: utf-8 -*-
print(__doc__)


import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

from Network import DNN
from Online_updating import Belief_Updating
from collections import Counter  
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score  



#s = pd.read_csv('data\\sushi3b.5000.10.score',  sep=' ', header=None, encoding='utf8')

data = "sushi"
#data = "movielens"
if data == "sushi":
    #Pre-processing
    #Read the data
    idata = pd.read_csv("sushi3.idata", sep='\t', encoding='utf8')
    score = pd.read_csv('sushi3b.5000.10.score',  sep=' ', header=None, encoding='utf8')
    score = score.values
    #Pick out goal data and do normalization
    icolumn = ["style","major group","minor group","heaviness","consumption frequency","normalized price","sell frequency"]
    idata = scale(idata[icolumn])
    pca = PCA(n_components=50, svd_solver='full')
    test = score[int(score.shape[0]*0.9):,:]


if data == "movielens":
#    #Pre-processing
#    #Read the data
    idata = pd.read_csv("genome-scores.csv", sep=',', encoding='utf8').values
    idata = idata[:1128*9000,2].reshape(-1,1128)
#    movie_number = 9000 #The maximum figure is 27278
#    score = pd.read_csv('ratings.csv',  sep=',', encoding='utf8')
#    score = score[score['movieId']<=movie_number]
#    
#    sparse = np.array(range(1,movie_number+1)).reshape(-1,1)
##    sparse = np.tile(sparse,(7120,1))
#    sparse = DataFrame(sparse, columns=["movieId"])
#    
#    start = 0
#    score_old =  pd.merge(sparse, score[score['userId']==1], on='movieId', how='outer')
#    score_old['rating'] = score_old['rating'].replace(np.nan, 0)
##    for batch in range(sparse.shape[0]//10000):7120
#    for user in range(7120-1):
#        score_batch = score[score['userId']==user+2]
#        score_new = pd.merge(sparse, score_batch, on='movieId', how='outer')
#        score_new['rating'] = score_new['rating'].replace(np.nan, 0)
#        score_old = score_old.append(score_new)
#        print (user)
#    score_old.to_csv ("MovieLens_user.csv" , encoding = "utf-8")　
    score = pd.read_csv('Movie_score.csv',  encoding='utf8').values[:,1:]
    pca = PCA(n_components=1400, svd_solver='full')
    test = score[int(score.shape[0]*0.9):,:]


pca.fit(score)                 
pca_score = pca.transform(score)
#print(pca.explained_variance_ratio_)  

x = pca_score
clusters=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
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
plt.title("Sushi dataset")
plt.savefig("sushi")
plt.show()

#Key represents the usetype, value represents the score
usertype = {}
#Key represents the usetype, value represents the item feature
itemtype = {}
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


#The type of the users
utype = userkmeans.labels_
#Split users into different groups
for i in range(user_cluster):
    usertype[i] = score[utype == i, :]
#Concate the item features and the ratings
if data == "sushi":
    for j in range(user_cluster):
        itemtype[j] = usertype[j].reshape(-1,1)
        itemtype[j] = np.concatenate([np.tile(idata,(usertype[j].shape[0],1)), itemtype[j]],axis=1)
        itemtype[j] = itemtype[j][itemtype[j][:,-1]!=-1,:]
        
if data == "movielens":
#for j in range(user_cluster):
#    itemtype[j] = usertype[j].reshape(-1,1)
#    batch_itemtype_1 = itemtype[j][0:itemtype[j].shape[0]//10,:]
#    itemtype[j] = np.zeros((1,idata.shape[1]))
#    for batch in range(len(itemtype[j]//10)):
#        batch_itemtype_2 = itemtype[j][0:itemtype[j].shape[0]//10*(batch+2),:]
#        batch_itemtype_2 = np.concatenate([np.tile(idata,(itemtype[j].shape[0]//10,1)),batch_itemtype_2],axis=0)
#        itemtype[j] = np.concatenate([batch_itemtype_1, batch_itemtype_2], axis=1)
#        itemtype[j] = itemtype[j][1:,:]
#        batch_itemtype_1 = batch_itemtype_2
#    itemtype[j] = itemtype[j][itemtype[j][:,-1]!=-1,:]

#    for j in range(user_cluster):
#        itemtype[j] = usertype[j].reshape(-1,1)
#        item_stack = np.zeros((1,idata.shape[1]+1))
#        for batch in range(itemtype[j].shape[0]//100000):
#            index_1 = (itemtype[j].shape[0]//100000)*batch
#            index_2 = (itemtype[j].shape[0]//100000)*(batch+1)
#            batch_itemtype = np.tile(idata,(index_2-index_1,1))
#            batch_itemtype = np.concatenate([batch_itemtype,itemtype[j][index_1:index_2,:]],axis=0)
#            item_stack = np.concatenate([batch_itemtype,index_1], axis=1)
#        itemtype[j] = item_stack[1:,:]
#        itemtype[j] = itemtype[j][itemtype[j][:,-1]!=0,:]
    
    for j in range(user_cluster):
        itemtype[j] = usertype[j].reshape(-1,1)
        item_stack = np.concatenate([idata,usertype[j][0,:].reshape(-1,1)], axis=1)
        for batch in range(usertype[0].shape[0]):
            next_batch = np.concatenate([idata,usertype[j][batch+1,:].reshape(-1,1)], axis=1)
            next_batch = next_batch[next_batch[:,-1]!=0,:]
            item_stack = np.concatenate([item_stack,next_batch],axis=0)
        itemtype[j] = item_stack
#        itemtype[j] = itemtype[j][itemtype[j][:,-1]!=0,:]


#Deep Neural Networks
item = []
for model in range(len(itemtype)):
    DNN_model = DNN(batch_size=50, learning_rate=0.01, training_epochs=20, dropout_rate=0.0,
                      batch_norm_use=False)
    output = DNN_model(training_data= itemtype[model] ,item=idata , display_step = 1,n_input = 7, n_output = 1)
    item.append(output)


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

if data == "sushi":
    cluster = [9,8,9,8,6,8,7,5]
    
if data == "movielens":
    cluster = [9,8,9,8,6,8,7,5]
    
    
itemkmeans = {}
for m in range(len(cluster)):
    itemkmeans[m] = KMeans(n_clusters=cluster[m], random_state=0).fit(item[m])
#    itemcount[m] = Counter(itemkmeans[m].labels_)    
    label_pred = itemkmeans[m].labels_ #获取聚类标签
    centroids = itemkmeans[m].cluster_centers_ #获取聚类中心
    inertia = itemkmeans[m].inertia_ # 获取聚类准则的总和
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    #这里'or'代表中的'o'代表画圈，'r'代表颜色为红色，后面的依次类推
    j = 0 
    for i in label_pred:
        plt.plot([item[m][j:j+1,0]], [item[m][j:j+1,1]], mark[i], markersize = 5)
        j +=1
    plt.show()

output = {}
#store each model dictionary, the key and value of the dictionary are mean and variance
model = []
if data == "sushi":
    for i in range(len(cluster)):
        output[i] = itemkmeans[i].labels_.reshape(-1,1)
        output[i] = np.concatenate([np.array(range(0,100)).reshape(-1,1), output[i]],axis=1)
        np.savetxt('sushi_space'+str(i)+'.csv', output[i], delimiter = ',')
        model.append([])
        
if data == "movielens":
    for i in range(len(cluster)):
        output[i] = itemkmeans[i].labels_.reshape(-1,1)
        output[i] = np.concatenate([np.array(range(0,100)).reshape(-1,1), output[i]],axis=1)
        np.savetxt('movie_space'+str(i)+'.csv', output[i], delimiter = ',')
        model.append([])


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
#            pred_rating = current_model_mean[candidate_class]
            pred_rating = np.max(test[time_step,:])
            true_rating = test[time_step,column]
            loss = abs(pred_rating - true_rating)
            total_loss = loss + total_loss
            averge_loss = total_loss/iteration
            print ("Avg Loss:",averge_loss)
    

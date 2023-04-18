import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as m
import sys
import os
from scipy.stats import *
import seaborn as sns


x = sys.path[0].split("/")
l = len(x)
last = x[l-1]
newpath = sys.path[0].rstrip(last)
sys.path.append(newpath)
sys.path.append(newpath + "data/example-code/")


""" followed this tutorial: https://www.youtube.com/watch?v=JNlEIEwe-Cg """


def normalize_features(df):
    new_df = df.copy()
    means = df.mean()
    max_val = df.max()
    min_val = df.min()
    for col in df.columns:
        new_df[col] = (df[col] - means[col]) /  (max_val[col] - min_val[col])
    return new_df

def E_step(df, K, mix, means, covariance_mtx):
    weights = np.zeros((len(df), K))

    for cluster in range(K):
        i = 0
        for x_val in df.iterrows():
            weights[i, cluster] = mix[cluster] * multivariate_normal.pdf(x_val[1], mean= means[cluster], cov=covariance_mtx[cluster])
            i +=1
            
    den = np.sum(weights, axis=1) #this might be wrong
    weights = weights / (den.reshape((den.shape[0], 1)))

    return weights

def M_step(df, K, weights, means, covariance_mtx):
    (left, right) = zip(*weights)
    new_means = np.zeros(means.shape)
    new_covar_mtx = np.zeros(covariance_mtx.shape)
    data = df.to_numpy()
    for cluster in range(K):
        mu = sum(w * d / np.sum(weights[:,cluster]) for (w, d) in zip(weights[:,cluster], df.to_numpy()))
        new_means[cluster] = mu
        covar_weights = np.repeat(weights[:,cluster], 2*2).reshape(len(df), 2, 2) 
        diff = data - mu
        vertical = diff.reshape(diff.shape[0],diff.shape[1],1) #need to reshape to square the diff matrix using np.matmul
        horizontal = diff.reshape(diff.shape[0],1,diff.shape[1])

        new_covar_mtx[cluster] = np.sum(np.multiply(covar_weights, np.matmul(vertical , horizontal) ), axis=0) / np.sum(weights[:,cluster]) #this might be wrong

    new_mix = np.sum(weights, axis=0) / len(df)
    return new_mix, new_means, new_covar_mtx

def draw_clusters(weights, means, covariance_mtx, df, mix, problem, iteration):
    cluster1 = []
    cluster2 = []
    i = 0
    for w in weights:
        if(w[0] >= w[1]):
            cluster1.append(df.iloc[i].to_list())
        else:
            cluster2.append(df.iloc[i].to_list())
        i+=1
    cluster1 = np.asarray(cluster1)
    cluster2 = np.asarray(cluster2)
    plt.scatter(cluster1[:,0], cluster1[:,1], c='g', label='First cluster')
    plt.scatter(cluster2[:,0], cluster2[:,1], c='r', label='Second cluster')
    plt.legend()
    plt.ylabel('second column')
    plt.xlabel('first column')
    plt.title(problem + f" iteration = {iteration}")
    plt.savefig('/Users/jakehirst/Desktop/Prob ML/hw4/images/' + problem.replace(" ",  "_") + f"iteration_{iteration}.png")
    plt.close()
    return

df = pd.read_csv("/Users/jakehirst/Desktop/Prob ML/hw4/data/faithful/faithful.txt", header=None, delimiter=' ', names =['col0', 'col1'])
df = normalize_features(df)
# sns.histplot(df['col1'], bins=20, kde=False)
# sns.histplot(df['col0'], bins=20, kde=False)

problem = 'problem 3a'
K = 2
means = np.array([[-1,1], 
                  [1,-1]])
covariance_mtx = np.array([[[0.1, 0],
                           [0, 0.1]],
                          [[0.1, 0],
                           [0, 0.1]]])

mix = np.array([0.5, 0.5])
for i in range(1,101):
    weights = E_step(df, K, mix, means, covariance_mtx)
    mix, means, covariance_mtx = M_step(df, K, weights, means, covariance_mtx)
    if(i == 1 or i == 2 or i == 5 or i ==100):
        draw_clusters(weights, means, covariance_mtx, df, mix, problem, i)      
        
        
problem = 'problem 3b'
K = 2
means = np.array([[-1,-1], 
                  [1,1]])
covariance_mtx = np.array([[[0.5, 0],
                           [0, 0.5]],
                          [[0.5, 0],
                           [0, 0.5]]])
mix = np.array([0.5, 0.5])

for i in range(1,101):
    weights = E_step(df, K, mix, means, covariance_mtx)
    mix, means, covariance_mtx = M_step(df, K, weights, means, covariance_mtx)
    if(i == 1 or i == 2 or i == 5 or i ==100):
        draw_clusters(weights, means, covariance_mtx, df, mix, problem, i)      
        
        
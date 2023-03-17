import csv
import numpy as np
import scipy.stats
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import norm
import argparse
import pandas as pd
import os
import math as m
from numpy.linalg import inv
LOGISTIC = True
NONZERONUM = 1e-13 #need this for logs otherwise we get divide by zero and log(0)
np.random.seed(0)
DATA_FOLDER = os.getcwd() + "/hw2/bank-note/"
#DATA_FOLDER = "/Users/jakehirst/Desktop/Prob ML/hw2/bank-note/"

def get_data(file):
    filepath = DATA_FOLDER + file
    data = pd.read_csv(filepath, header=None)
    data.columns = ["variance", "skewness", "curtosis", "entropy", "label: genuine or forged"]
    features = data.drop("label: genuine or forged", axis=1)
    labels = data["label: genuine or forged"]
    return features, labels #can turn these into numpy arrays with np.asarray()

""" calls the logistic function, also known as the sigmoid function"""
def logistic(x,w):
    if(LOGISTIC):
        hofx = 1 / (1 + np.exp(-np.matmul(x,w.T)))
        hofx = hofx.reshape(len(hofx),1)
    else:#the only other method is probit for this question
        hofx = norm.cdf(np.matmul(x, w.T)) #derivative of cdf is pdf according to https://www.statology.org/cdf-vs-pdf/
        #𝑃(𝑌=1|𝑋)=Φ(𝜂) = ∫𝑒^(−𝑧2/2) / sqrt(2𝜋)𝑑𝑧
        hofx = hofx.reshape(len(hofx),1)
    return hofx

def get_likelihood(x,w,y):
    hofx = logistic(x,w)
    likelihood = np.power(hofx, y) * (np.power((1-hofx), (1-y)))
    
    return likelihood

def get_log_likelihood(x,w,y):
    hofx = logistic(x,w)
    hofx[hofx<NONZERONUM] = NONZERONUM #trying to add a nonzero term to prevent log(0) 
    hofx[hofx >= 1.0] = 1.0 - NONZERONUM #trying to add a nonzero term to prevent log(0) 
    #l(θ)=∑ yilog(hθ(xi)) + (1−yi)log(1−hθ(xi)) FORMULA
    left_side = y * np.log(hofx)
    right_side = (1 - y) * np.log(1-hofx)
    #print(left_side)
    #print(right_side)
    return np.sum(left_side + right_side)

"""gets first derivative for newton raphson method """
def get_1_gradient(x,w,y):
    if(LOGISTIC):
        hofx = logistic(x,w)
        gradient = np.matmul(np.transpose(x), (hofx - y)) + w
        gradient = gradient.T[0]
        return gradient.reshape(1, len(gradient))
    else:
        hofx = logistic(x,w)
        gradient = None
        
        return

"""had to rewrite the gradient function because idk how to put x and y in when optimizing with scipy """
def get_1_gradient_for_optimization(w):
    print(f"w = {w}")
    hofx = logistic(X,w) + NONZERONUM #trying to add a nonzero term to prevent log(0) 
    phi = np.matmul(X,w).reshape(hofx.shape)
    gradient_of_hofx = norm.pdf(phi,0,1)#gradient of a cdf is a pdf
    
    gradient = np.sum((- Y*(1/(hofx)) + (1-Y)*(1/(1-hofx)))*gradient_of_hofx * X, axis = 0) 
    print(f"gradient = {gradient}")
    return gradient
    
def get_log_likelihood_for_optimization(w): 
    hofx = logistic(X,w)
    #l(θ)=∑ yilog(hθ(xi)) + (1−yi)log(1−hθ(xi))
    left_side = Y * np.log(hofx + NONZERONUM)
    right_side = (1 - Y) * np.log(1-hofx + NONZERONUM)
    loglikelihood = -np.sum(left_side + right_side)
    print(f"negative log likelihood = {loglikelihood}")
    return loglikelihood
    

""" gets the second derivative (aka the Hessian) for Newton Raphson method """
def get_2_gradient(x,w,y):
    hofx = logistic(x,w)
    hofx[hofx<NONZERONUM] = NONZERONUM
    hofx[hofx >= 1.0] = 1.0 - NONZERONUM
    num_examples, num_dimensions = x.shape
    R = np.zeros((num_examples, num_examples))
    for n in range(num_examples):
        R[n,n] = hofx[n] * (1-hofx[n])
    
    hessian = np.matmul(np.matmul(x.T, R), x) 
    return hessian

def make_predictions(x,w,y):
    hofx = logistic(x,w)
    predictions = np.round(hofx)
    correct = np.sum(predictions == y)
    accuracy=correct / len(x)
    print(f"accuracy = {accuracy} number correct = {correct}")
    return accuracy, correct
    
    
def part_a(starting_w):
    LOGISTIC = True
    max_iterations = 100
    tolerance = 1e-5
    train_features, train_labels = get_data("train.csv")
    test_features, test_labels = get_data("test.csv")
    train_features["bias"] = 1 #adding a column of ones to the features as a bias term
    test_features["bias"] = 1 #adding a column of ones to the features as a bias term

    if(starting_w =="zeros"):
        w = np.zeros(train_features.shape[1])
    elif(starting_w == "random"):
        w = np.random.rand(train_features.shape[1])  
    
    test_x = np.asarray(test_features)
    test_y = np.asarray(test_labels, dtype="float64")
    test_y = test_y.reshape(len(test_y), 1)
    
    x = np.asarray(train_features)
    y = np.asarray(train_labels, dtype="float64")
    w = w.reshape(1,len(w))
    y = y.reshape(len(y), 1)
    likelihood = get_likelihood(x, w, y)
    log_likelihood = get_log_likelihood(x, w, y)
    print("\nSTARTING ACCURACY = ")
    make_predictions(x,w,y)
    print("\n")

    """newton raphson method"""
    for i in range(max_iterations):
        first_grad = get_1_gradient(x,w,y) 
        hessian = get_2_gradient(x,w,y) 
        wold = w
        w = w - (np.matmul(inv(hessian), first_grad.T)).T
        if(abs(np.linalg.norm(w - wold)) < tolerance):
            print("converged")
        log_likelihood = get_log_likelihood(x, w, y)
        print(f"w = {w}")
        #print(f"hessian = \n{hessian}")
        #print(f"first_grad = {first_grad}")
        print(f"i = {i} log_likelihood = {log_likelihood}")
        if(abs(np.linalg.norm(w - wold)) < tolerance):
            print("converged")
            print("test accuracy:")
            make_predictions(test_x,w,test_y)
            break
        make_predictions(x,w,y)
    

    return

def part_b(starting_w):
    global LOGISTIC
    global X
    global Y
    LOGISTIC = False
    max_iterations = 100
    tolerance = 1e-5
    train_features, train_labels = get_data("train.csv")
    test_features, test_labels = get_data("test.csv")
    train_features["bias"] = 1 #adding a column of ones to the features as a bias term
    test_features["bias"] = 1 #adding a column of ones to the features as a bias term

    if(starting_w =="zeros"):
        w = np.zeros(train_features.shape[1])
    elif(starting_w == "random"):
        w = np.random.rand(train_features.shape[1])    
        
    test_x = np.asarray(test_features)
    test_y = np.asarray(test_labels, dtype="float64")
    test_y = test_y.reshape(len(test_y), 1)        

    x = np.asarray(train_features)
    y = np.asarray(train_labels, dtype="float64")
    w = w.reshape(1,len(w))
    y = y.reshape(len(y), 1)
    X = x
    Y = y

    optimized = minimize(get_log_likelihood_for_optimization, w, method='BFGS', jac=get_1_gradient_for_optimization)
    print("training:")
    make_predictions(x,optimized.x,y)
    print("test:")
    make_predictions(test_x, optimized.x, test_y)
    return

part_a("zeros")
part_a("random")
part_b("zeros")
part_b("random")

    
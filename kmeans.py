import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import copy
import sys

# function to calculate the squared distance
def compute_distance(a, b):
    return np.sqrt(np.sum(np.power((a - b), 2)))

# initialize centroids function to pick unique random datapoints as centroids
def initialize_centroids(a, X,y):
    centroids = []
    if a==3:
        d = []
        while len(centroids) < a:
            i = np.random.randint(0, len(X))
            f = 0
            for k in d:
                if y[i]==y[k]:
                    f=1
                    break
            if f==0:
                d.append(i)
                centroids.append(X[i])
        i = 0
        for k in d:
            y[y==y[k]] = i
            i+=1
    else:    
        while len(centroids) < a:
            i = np.random.randint(0, len(X))
            if list(X[i]) not in centroids:
                centroids.append(list(X[i]))
    centroids = np.array(centroids)
    return centroids

# assigning each datapoint to nearest cluster
def perform_clustering(centers, centroids, X):
    centers_X = {}
    for i in range(len(X)):
        centers_X[i] = (100000000, 0)
        for j in range(centers):
            dist = compute_distance(X[i], centroids[j])
            if dist <= centers_X[i][0]:
                centers_X[i] = (dist, j)
    return centers_X

# updating the centroids as mean of datapoints in each cluster
def update_centroids(centroids, centers_X, X):
    centroids = np.zeros(centroids.shape)
    counts = {}
    for k, v in centers_X.items():
        centroids[v[1]] += X[k]
        if v[1] in counts:
            counts[v[1]] += 1
        else:
            counts[v[1]] = 1
    for i in range(len(centroids)):
        if i in counts:
            centroids[i] /= counts[i]
    return centroids

# function to compute accuracy for 3 centroids clustering
def compute_accuracy(centers_X,y):
    acc = 0
    for k, v in centers_X.items():
        if v[1] == y[k]:
            acc += 1
    acc = acc / len(centers_X)
    return acc*100

# within cluster sum of squares distance cost function
def wcss(centers_X):
    c = 0
    for k,v in centers_X.items():
        c += v[0]**2
    return c
    
# elbow method implementation to determine the number of clusters
def elbow_method(X,y):
    cost = []
    print("Elbow-Method")
    for centers in range(1,10):     
        centroids = initialize_centroids(centers, X,y)    
        for i in range(100):    
            centers_X = perform_clustering(centers, centroids, X)
            centroids = update_centroids(centroids, centers_X, X)
        cost.append(wcss(centers_X))
        print("Iteration: ", centers)
    plt.figure(1)
    plt.plot(np.linspace(1,len(cost),len(cost)),cost)
    plt.xlabel("Iterations")
    plt.ylabel("Within Clusters Sum of Squares")
    plt.title("Elbow method for K-means clusturing")
    plt.show()

def avg_silhoutte(centers_X,X,y):
    s = np.zeros(len(X))
    
    
def silhoutte_score(X,y):
    cost = []
    print("Silhoutte_score")
    for centers in range(1,10):     
        centroids = initialize_centroids(centers, X,y)    
        for i in range(100):    
            centers_X = perform_clustering(centers, centroids, X)
            centroids = update_centroids(centroids, centers_X, X)
        cost.append(avg_silhoutte(centers_X,X,y))
        print("Iteration: ", centers)
    plt.figure(1)
    plt.plot(np.linspace(1,len(cost),len(cost)),cost)
    plt.xlabel("Iterations")
    plt.ylabel("Silhoutte score")
    plt.title("Silhoutte score vs Iterations")
    plt.show()    
    
def main():
    if len(sys.argv)<2:
        print("Please input Y/N for elbow method as 1st argument")
        print("Please input k value as 2nd argument")
        exit(0)
    elif len(sys.argv)<3:	
        print("Please input k value")
        exit(0)
    data = pd.read_csv("iris.data", header=None).sample(frac=1)
    X = data.iloc[:, :-1].values
    y1 = data.iloc[:, -1].values
    # apply elbow method
    if sys.argv[1].lower()=="y":
        elbow_method(X,copy(y1))
    centers = int(sys.argv[2])
    acc = []
    wcs_cost = []
    print("K-means")
    for _ in range(10):
        y = copy(y1)
        centroids = initialize_centroids(centers, X,y)
        cost = []
        for i in range(10):
            centers_X = perform_clustering(centers, centroids, X)
            centroids = update_centroids(centroids, centers_X, X)
            cost.append(wcss(centers_X))
        wcs_cost.append(wcss(centers_X))
        acc.append(compute_accuracy(centers_X,y))        
        print("Iteration: ", _+1)
    plt.figure(3)
    plt.plot(np.linspace(1,len(cost),len(cost)),cost)
    plt.xlabel("Iterations")
    plt.ylabel("Within Clusters Sum of Squares")
    plt.title("Within Cluster squared distance cost for "+ str(centers)+ " clustures")
    if centers==3:
        print("Average Accuracy: ", np.mean(acc), "%")
    print("Average Within cluster sum of sqaures distance: ",np.mean(wcs_cost))
    print("Centroids")
    print(centroids)
    plt.show()
    
if __name__ == "__main__":
    main()

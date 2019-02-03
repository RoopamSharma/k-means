Problem:
The problem here is to cluster image data using the K-means unsupervised learning method.
We can evaluate clustering accuracy for the number of clusters as 3.
Data:
The data has 4 dependent variables, sepal-length, sepal-width, petal-length, petal-width and class labels.
Data is stored in iris.data file.
Method (Program Structure)  and Results:
First we calculate the Sum of squares distance for different k values from 1 through 9 to determine the optimal k value. I have used the elbow method to determine the number of clusters.
The aim is to have minimum within cluster sum of squares distance while keeping the number of clusters less. Since, with the increase in number of clusters the distance will approach 0, apparently when number of clusters are equal to the no. of data points. So, elbow is the point that gives the optimal k value.
Here we got K-value as 3.
Algorithm steps:
The program starts by initializing the clusters centroids by choosing unique random datapoint from the dataset.
Then we find for every datapoint the cluster centroid it belongs to by computing the distance between the point and the centroid, then it is assigned to the minimum distant cluster.
Next step is to update the centroids by taking the average of the datapoints that belongs to each cluster.
The process is repeated until convergence.
If we repeat the full K-means algorithm multiple times, we can compute the average accuracy on 3 centroids clustering and I got maximum 88% accuracy. And average accuracy of 85%.
We can also compute the within cluster sum of squares distance cost to analyze how our model is performing. This cost is computed after each iteration and it shows how the cost converges. 
Running the program:
The program requires the K value and Y/N flag for elbow method as command line input.
It can be run by executing the command:
    python kmeans.py Y k		 	for executing elbow method
    python kmeans.py N k 		for not executing elbow method
K can be any value for number of clusters.

import random as rd
import numpy as np
from scipy.spatial.distance import cosine

# class to implemenet k-means clustering with cosine distance (and faster than sklearn's implementation)
class Kmeans(object):
    ''' implementation of k-means that is clean, fast, and can use cosine distance 
        TODO: kmeans++ '''
    def __init__(self, k, max_iters=100, metric='cosine'):
        ''' k = number of clusters. 
        max_iters = maximum # of iterations '''
        self.k = k
        self.max_iters = max_iters
        self.iters = 0
        self.centroids = None
        self.converged = None
        self.metric = metric

    def fit(self, X):
        ''' X should be a numpy array where rows are each vectors '''
        self.converged = False
        self.centroids = np.array(rd.sample(X, self.k))
        self.iters = 0
        for _ in xrange(self.max_iters):
            self.iters += 1
            distances = self.calc_distances(X)
            new_centroids = self.compute_centroids(X, distances)
            if (new_centroids == self.centroids).all():
                self.converged = True
                break
            self.centroids = new_centroids

    def calc_distances(self, X):
        ''' calc a k x m matrix with the distances from each centroid to each X vector '''
        return map(lambda c: self.calc_distance_for_centroid(c, X), self.centroids)

    def calc_distance_for_centroid(self, centroid, X):
        ''' calculate the distance for each vector to the given centroid '''
        if self.metric == 'cosine':
            return 1 - (np.dot(X,centroid)/np.linalg.norm(X,axis=1))/np.linalg.norm(centroid)
        else:
            return 

    def compute_centroids(self, X, distances):
        ''' compute the newest centroids '''
        ''' b is a binary array which is False by default and 
            True if the centroid is closest to vector '''
        b = distances == np.min(distances, axis=0)
        ''' kind of confusing below:
            np.dot(b,X) returns the sum of all vectors which are closest to each centroid.
            b.sum(axis=1) is the number of vectors which are closest to each centroid.
            we divide to get the average. '''
        return (1. * np.dot(b, X) / b.sum(axis=1).reshape(-1, 1))

    def compute_sse(self, X):
        ''' compute standard squared error '''
        distances = self.calc_distances(X)
        min_dist = np.min(distances, axis=0)
        return np.dot(min_dist, min_dist.T)

    def get_labels(self, X):
        ''' get which centroid is closest to each vector '''
        distances = self.calc_distances(X)
        return np.argmin(distances, axis=0)

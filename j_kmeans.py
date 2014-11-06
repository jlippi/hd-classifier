import random as rd
import numpy as np
from scipy.spatial.distance import cosine


class Kmeans(object):

    def __init__(self, k, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.converged = None

    def fit(self, X):
        self.converged = False
        self.centroids = np.array(rd.sample(X, self.k))
        for _ in xrange(self.max_iters):
            distances = self.calc_distances(X)
            new_centroids = self.compute_centroids(X, distances)
            if (new_centroids == self.centroids).all():
                self.converged = True
                break
            self.centroids = new_centroids

    def calc_distances(self, X):
        return map(lambda c: self.calc_distance_for_centroid(c, X), self.centroids)

    def calc_distance_for_centroid(self, centroid, X):
        return 1 - (np.dot(X,centroid)/np.linalg.norm(X,axis=1))/np.linalg.norm(centroid)

    def compute_centroids(self, X, distances):
        b = distances == np.min(distances, axis=0)
        return (1. * np.dot(b, X) / b.sum(axis=1).reshape(-1, 1))

    def compute_sse(self, X):
        distances = self.calc_distances(X)
        min_dist = np.min(distances, axis=0)
        return np.dot(min_dist, min_dist.T)

    def get_labels(self, X):
        distances = self.calc_distances(X)
        return np.argmin(distances, axis=0)

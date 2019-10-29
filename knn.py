import numpy as np


class KNeighborsClassifier:

    def __init__(self, k):
        self.k = k
        self.X = None
        self.y = None

    def distance(self, x, X):
    	return np.sqrt(np.sum((x-X)**2, axis=1))

    def top_k(self, X):
    	min_distances = np.zeros((len(X), self.k))

    	for i, row in enumerate(X):
    		min_distances[i, :] = np.argsort(row)[-self.k:]

    	return min_distances

    def weighted_voting(self, ):
    	pass

    def fit(self, X_train, y_train):
        self.X = np.array(X_train)
        self.y = np.array(y_train)

    def predict(self, X_test):
    	predictions = np.zeros(len(X_test))
    	distances = np.zeros((len(X_test), len(self.X)))

    	for i, element in enumerate(X_test):
    		distances[i, :] = self.distance(element, self.X)

    	closest_points = self.top_k(distances)
    	votes = np.zeros(closest_points.shape)

    	votes = self.y[np.array(closest_points, dtype=int)]

    	for i, row in enumerate(votes):
    		(values, counts) = np.unique(votes, return_counts=True)
    		predictions[i] = row[np.argmax(counts)]

    	return predictions

    def score(self, predictions, y_test):
        results = np.where(predictions == y_test, 1, 0)
        
        return (1 - (np.sum(results) / len(predictions)))





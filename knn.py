import numpy as np


class KNeighborsClassifier:

	def __init__(self, k):
		self.k = k
		self.X = None
		self.y = None

	def distance(self, x, X):
		"""
			Euclidean distance as the distance metric between points
		"""
		return np.sqrt(np.sum((x-X)**2, axis=1))

	def top_k(self, X):
		"""
			Return the k closest neighbors
		"""
		min_distances = np.zeros((len(X), self.k))

		for i, row in enumerate(X):
			min_distances[i, :] = np.argsort(row)[-self.k:]

		return min_distances

	def fit(self, X_train, y_train):
		"""
			Initialize the global variables X and y
		"""
		self.X = np.array(X_train)
		self.y = np.array(y_train)

	def predict(self, X_test):
		predictions = np.zeros(len(X_test))
		distances = np.zeros((len(X_test), len(self.X)))

		# for every point of X_test
		# calculate its distance to every point of X_train
		for i, element in enumerate(X_test):
			distances[i, :] = self.distance(element, self.X)

		# get the k nearest neighbors
		closest_points = self.top_k(distances)

		# get the classes of the k nearest neighbor
		votes = np.zeros(closest_points.shape)
		votes = self.y[np.array(closest_points, dtype=int)]
		
		# use majority voting to classify each datapoint of X_test
		for i, row in enumerate(votes):
			(values, counts) = np.unique(row, return_counts=True)
			predictions[i] = row[np.argmax(counts)]

		return predictions

	def score(self, predictions, y_test):
		"""
			Compare predictions with actual labels
			and return the accuracy of the predictions
		"""
		# create a vector with 1 in the indices where the prediction matches the actual class
		# and 0 everywhere they don't match
		results = np.where(predictions == y_test, 1, 0)
		
		# add the elements of the results vector to get the count of the correct predictions (TP + TN)
		return (1 - (np.sum(results) / len(predictions)))





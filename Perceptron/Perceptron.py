import numpy as np

class Perceptron(object):

	# Parameters
	eta : float
	n_iter : int
	random_state : int

	# Attributes
	w_ : list
	errors_ : list

	# Constructor method with three parameters 
	def __init__(self, eta = 0.01, n_iter=50, random_state=1):
		self.eta = eta
		self.n_iter = n_iter
		self.random_state = random_state

	# Setter method 'fit' fits training data with independent and dependent variables as parameters
	def fit(self, X, y):
		#Fit training data
		self : object
		rgen = np.random.RandomState(self.random_state)
		self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

		self.errors_ = []

		for _ in range(self.n_iter):
			
			errors = 0

			for xi, target in zip(X, y):

				update = self.eta * (target - self.predict(xi))
				self.w_[1:] += update * xi
				self.w_[0] += update
				errors += int(update != 0.0)

			self.errors_.append(errors)

		return self 

	# Setter method used to update weights 'w_'
	def net_input(self, X):
		return np.dot(X, self.w_[1:]) + self.w_[0]

	# Getter method used to return model with best weights
	def predict(self, X):
		return np.where(self.net_input(X) >= 0.0, 1, -1)

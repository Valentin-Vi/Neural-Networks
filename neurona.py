import numpy as np

def sigmoid(x):
    # Funcion de activacion de Sigmoid: f(x) = 1 / (1 + e^(-x))
	return 1 / (1 + np.exp(-x))

class Neuron:
	def __init__(self, weights, bias):
		self.w = weights
		self.b = bias

	def feedforward(self, x):
        # Weighting inputs
		z = np.dot(self.w, x) + self.b # z = 7
		return sigmoid(z)

w = np.array([0, 1])    # w1 = 0, w2 = 1
b = 4                   # b = 4

n = Neuron(w, b)

x = np.array([2, 3])    # x1 = 2, x2 = 3

print(n.feedforward(x)) # 0.9990889488055994

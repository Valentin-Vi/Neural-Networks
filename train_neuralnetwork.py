import numpy as np

'''
  Funcion de Activacion:
  Sigmoid -> f(x)
'''
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

'''
  Derivada de Funcion Activacion de Sigmoid
  f'(x)
'''
def d_sigmoid(x):
  fx = sigmoid(x)
  return fx * (1 - fx)

'''
  Funcion de Loss:
  MSE (Mean Square Error) -> MSE(Ypred) 
'''
def mse_loss(y_true, y_pred):
  return ((y_true - y_pred) ** 2).mean()

class NeuralNetwork:
  def __init__(self):
    '''
      Inicializando weights y biases aleatoriamente.
      h1 Є w1 ∧ w2 ∧ b1
      h2 Є w3 ∧ w4 ∧ b2
      o1 Є w5 ∧ w6 ∧ b3
    '''
    # Weights
    self.w1 = np.random.normal()
    self.w2 = np.random.normal()
    self.w3 = np.random.normal()
    self.w4 = np.random.normal()
    self.w5 = np.random.normal()
    self.w6 = np.random.normal()

    # Biases
    self.b1 = np.random.normal()
    self.b2 = np.random.normal()
    self.b3 = np.random.normal()

  def feedforward(self, x):
    h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
    h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
    o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
    return o1

  def train(self, data, all_y_trues):
    learn_rate = 0.1
    epochs = 1000

    for epoch in range(epochs):
      for x, y_true in zip(data, all_y_trues):

        """
          -- Feedforward --
        """
        # Neurona h1
        sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
        h1 = sigmoid(sum_h1)
        # Neurona h2
        sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
        h2 = sigmoid()
        # Neurona o1
        sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
        y_pred = o1 = sigmoid(sum_o1)

        """
          -- Derivada Parcial --
          
          Donde d_X es derivada de X, X siendo una variable.
        """
        d_L_d_ypred = -2 * (y_true - y_pred) # dL/dYpred

        # Neurona o1
        d_ypred_d_w5 = h1 * d_sigmoid(sum_o1) # dYpred/dw5
        d_ypred_d_w6 = h2 * d_sigmoid(sum_o1) # dYpred/dw6
        d_ypred_d_b3 = d_sigmoid(sum_o1)  # dYpred/db3

        d_ypred_d_h1 = self.w5 * d_sigmoid(sum_o1) # dYpred/dh1
        d_ypred_d_h2 = self.w6 * d_sigmoid(sum_o1) # dYpred/h2

        # Neurona h1
        d_h1_d_w1 = x[0] * d_sigmoid(sum_h1)
        d_h1_d_w2 = x[1] * d_sigmoid(sum_h1)
        d_h1_d_b1 = d_sigmoid(sum_h1)

        # Neurona h2
        d_h2_d_w3 = x[0] * d_sigmoid(sum_h2)
        d_h2_d_w4 = x[1] * d_sigmoid(sum_h2)
        d_h2_d_b2 = d_sigmoid(sum_h2)

        """
          -- Descenso de Gradiente Estocástico --
        """
        # Neurona h1
        self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
        self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
        self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

        # Neurona h2
        self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
        self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
        self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

        # Neurona o1
        self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
        self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
        self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

      """
        -- Calculando Loss por cada Epoch --
      """
      if epoch % 10 == 0:
        y_preds = np.apply_along_axis(self.feedforward, 1, data)
        loss = mse_loss(all_y_trues, y_preds)
        print("Epoch %d loss: %.3f" % (epoch, loss))

'''
  Instanciando dataset.
'''
data = np.array([
  [-2, -1],
  [25, 6],
  [17, 4],
  [-15, -6],
])
all_y_trues = np.array([
  1,
  0,
  0,
  1,
])

network = NeuralNetwork()
network.train(data, all_y_trues)
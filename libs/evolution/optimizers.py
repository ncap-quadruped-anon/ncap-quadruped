import abc

import numpy as np


class Optimizer(abc.ABC):
  def initialize(self, params: np.ndarray) -> None:
    """Initialize the optimizer with the parameters `params`."""
    pass

  @abc.abstractmethod
  def step(self, grad: np.ndarray) -> np.ndarray:
    """Use the gradient `grad` to update and return the current parameters."""
    pass


class SGD(Optimizer):
  def __init__(self, learning_rate=0.1, momentum=0.):
    self.learning_rate = learning_rate
    self.momentum = momentum

  def initialize(self, params: np.ndarray):
    self.v = np.zeros_like(params)

  def step(self, grad: np.ndarray):
    self.v = self.learning_rate * grad + self.momentum * self.v
    return self.v


class Adam(Optimizer):
  def __init__(self, learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-08):
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon

  def initialize(self, params: np.ndarray):
    self.m = np.zeros_like(params)
    self.v = np.zeros_like(params)
    self.t = 0

  def step(self, grad: np.ndarray):
    self.t += 1
    a = self.learning_rate * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)
    self.m = self.beta1 * self.m + (1 - self.beta1) * grad
    self.v = self.beta2 * self.v + (1 - self.beta2) * grad * grad
    return a * self.m / (np.sqrt(self.v) + self.epsilon)

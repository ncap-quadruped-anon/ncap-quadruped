import torch


def identity(x):
  return x


def sigmoid(x):
  return torch.sigmoid(x)


def tanh(x):
  return torch.tanh(x)


def threshold(x):
  return torch.gt(x, 0.).float()


def hardsigmoid(x):
  return 0.5 * torch._C._nn.hardtanh(x, -1., 1.) + 0.5


def hardtanh(x):
  return torch._C._nn.hardtanh(x, -1., 1.)


def relu(x):
  return torch.relu(x)


def retanh(x):
  return torch.relu(torch.tanh(x))


def unitrelu(x):
  return torch.clamp(x, min=0, max=1)

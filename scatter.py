import torch as t
from torch import nn

class Scatter(nn.Module):
  def __init__(self, B, T, nx, ny):
    super(Scatter, self).__init__()
    self.T = T
    self.nx = nx
    self.ny = ny
    self.mus = t.empty((B, T, 2), requires_grad=True)

  def params(self):
    return self.mus

  def init(self):
    nn.init.uniform_(self.mus, a=0.0, b=1.0)
    with t.no_grad():
      self.mus[...,0] *= self.nx
      self.mus[...,1] *= self.ny

  def forward(self):
    return self.mus



   


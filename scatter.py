import torch as t
from torch import nn

class Scatter(nn.Module):
  def __init__(self, B, P, nx, ny):
    super(Scatter, self).__init__()
    self.P = P
    self.nx = nx
    self.ny = ny
    self.points = t.empty((B, P, 2), requires_grad=True)

  def init_points(self):
    nn.init.uniform_(self.points, a=0.0, b=1.0)
    with t.no_grad():
      self.points[...,0] *= self.nx
      self.points[...,1] *= self.ny

  def forward(self):
    return self.points



   


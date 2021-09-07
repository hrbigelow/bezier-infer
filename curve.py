import torch as t
from torch import nn
from math import factorial as fac

# Uses Bernstein Polynomial Form
# b_{v,n}(t) := (n-choose-v) t^v (1-t)^(n-v)

# Let P = [[x0, y0], [x1,y1], ..., [xn,yn]]
# Let n = len(P) - 1 
# Bezier(t,P) := sum_{v \in 0..n} { b_{v,n}(t) P[v] }

class BezierCurve(nn.Module):
  """Calculate the evaluation points of a Bezier"""
  def __init__(self, B, P, nx, ny, T, nr=10000):
    super(BezierCurve, self).__init__()
    self.P = P
    self.T = T
    self.points = t.empty((B, P, 2), requires_grad=True)
    self.nx = nx
    self.ny = ny
    self.nudge_reps = nr
    deg = P - 1
    self.coeff = t.tensor([fac(deg) / (fac(v) * fac(deg-v)) for v in range(P)])
    self.sub_coeff = t.tensor([fac(deg-1) / (fac(v) * fac(deg-v)) for v in
      range(P-1)])
    self.ls = t.linspace(0, 1, T).unsqueeze(0).repeat(B,1)
    self.init_poly()

  def params(self):
    return self.points

  def init_poly(self):
    # bernstein: T,P 
    vs = t.arange(self.P) # [0, P)
    vs_rev = t.flip(vs, dims=(0,))
    lsu = self.ls.unsqueeze(2)
    lsu_inv = 1.0 - lsu

    self.bernstein = self.coeff * lsu ** vs * lsu_inv ** vs_rev 

    sub_vs = vs[:-1]
    sub_vs_rev = t.flip(sub_vs, dims=(0,))

    self.sub_bernstein = self.sub_coeff * lsu ** sub_vs * lsu_inv ** sub_vs_rev 

  def init(self):
    nn.init.uniform_(self.points, a=0.0, b=1.0)
    self.ls[:] = t.linspace(0, 1, self.T)
    with t.no_grad():
      self.points[...,0] *= self.nx
      self.points[...,1] *= self.ny
    self.nudge_ls(self.nudge_reps)

  def set_points(self, points):
    with t.no_grad():
      self.points[:] = points
    self.nudge_ls(self.nudge_reps)

  def nudge_ls(self, reps):
    for _ in range(reps):
      with t.no_grad():
        # curve, _ = self.forward()
        curve = self.forward()
      diff = curve[:,1:,:] - curve[:,:-1,:] # B,T-1,2
      dist = t.einsum('bti, bti -> bt', diff, diff).sqrt()
      dif_dist = dist[:,1:] - dist[:,:-1] # B,T-2
      sum_dist = dist[:,1:] + dist[:,:-1]
      r = 0.5 * dif_dist / sum_dist # B,T-2
      flor = t.minimum(r, t.tensor(0.0)) 
      ceil = t.maximum(r, t.tensor(0.0))
      lsd = self.ls[:,1:] - self.ls[:,:-1] # B,T-1
      self.ls[:,1:-1] += ceil * lsd[:,1:] + flor * lsd[:,:-1]
      self.init_poly()


  def forward(self):
    # bernstein: T,P   points: B,P,2 
    # curve: B,T,2
    curve = self.bernstein @ self.points
    # with t.no_grad():
      # sub_points = self.P * (self.points[:,1:,:] - self.points[:,:-1,:])
    # curve_grad = self.sub_bernstein @ sub_points
    # return curve, curve_grad
    return curve


from base_model import BaseModel
from curve import BezierCurve
from mixture import Mixture
from scatter import Scatter

class Bezier(BaseModel):
  """Bezier Curve with individual Sigmas"""
  def __init__(self, P, B, T, nx, ny, sigma, single_sigma):
    super(Bezier, self).__init__()
    self.means = BezierCurve(B, P, nx, ny, T)
    self.mix = Mixture(B, T, nx, ny, sigma, single_sigma)

  def post_step_hook(self):
    self.means.nudge_ls(3)

class MuScatter(BaseModel):
  """Independent Mus and Sigmas"""
  def __init__(self, B, T, nx, ny, sigma, single_sigma):
    super(MuScatter, self).__init__()
    self.means = Scatter(B, T, nx, ny)
    self.mix = Mixture(B, T, nx, ny, sigma, single_sigma)

  def post_step_hook(self):
    pass 

import torch as t
from torch import nn, optim
import curve
import mixture
import scatter
import xent

class BezierModel(nn.Module):
  """The Generative model.  Produces the Gaussian Mixture for a rendered
  Bezier Curve"""

  def __init__(self, P, B, T, nx, ny, sigma, steps, sample_idx, report_every=1, print_callback=None):
    super(BezierModel, self).__init__()
    self.curve = curve.BezierCurve(B, P, nx, ny, T)
    self.scatter = scatter.Scatter(B, P, nx, ny)
    self.mix = mixture.Mixture(B, P, nx, ny, sigma)
    self.kldivloss = xent.KLDivLoss()
    self.steps = steps
    self.report_every = report_every
    self.sample_idx = sample_idx
    self.print_fn = print_callback
    self.pgmap = { 'points': 0, 'sigma': 1 }
 
  def init_points(self):
    # self.curve.init_points()
    self.scatter.init_points()

  def set_points(self, points):
    self.curve.set_points(points)

  def init_sigma(self, sigma):
    self.mix.init_sigma(sigma)

  def cuda(self):
    super(BezierModel, self).cuda()
    param_groups = [
        # { 'params': self.curve.points }
        { 'params': self.scatter.points },
        { 'params': self.mix.sigma },
        ]
    self.opt = optim.Adam(param_groups, lr=0.0)
    # self.opt = optim.SGD(param_groups, lr=0.0, momentum=0.8)

  def set_lr(self, group, lr):
    self.opt.param_groups[self.pgmap[group]]['lr'] = lr

  def get_lr(self, group):
    return self.opt.param_groups[self.pgmap[group]]['lr']

  def forward(self):
    pts = self.scatter() 
    mixture = self.mix(pts)
    return mixture

  def forward_bck(self):
    """Produce the data to be compared to the target"""
    curve, curve_grad = self.curve()
    mixture = self.mix(curve)
    return mixture

  def infer(self, trg_dist):
    """Does gradient descent on the points in the curve"""
    sched = {}
    sched['points'] = {0: 1e-1, 30000: 1e-3, 40000: 1e-4, 50000: 2e-6}
    sched['sigma'] = {0: 1e-1, 30000: 5e-4, 40000: 3e-4, 50000: 1e-4 }

    with t.no_grad():
      trg_dist = self.mix.process(trg_dist)
    trg_dist_log = t.where(trg_dist > 0.0, trg_dist.log(), t.zeros_like(trg_dist))
    trg_h = - (trg_dist * trg_dist_log).sum(dim=(1,2))

    for step in range(self.steps):

      self.opt.zero_grad()
      for k, v in self.pgmap.items():
        if step in sched[k]:
          self.set_lr(k, sched[k][step])

      current_mixture = self() # weird, but true
      l = self.kldivloss(trg_dist, current_mixture)
      l.sum().backward()
      self.opt.step()
      self.curve.nudge_ls(3)
      """
      """
      # Learning rate scheduling should be applied *after* optimizer's update
      # see https://pytorch.org/docs/stable/optim.html 'How to adjust Learning
      # Rate'

      if step % self.report_every == 0:
        # print(f'Step: {step}')
        sigma_lr = self.get_lr('sigma')
        points_lr = self.get_lr('points')
        kldiv = l - trg_h 
        print(
            f'Step: {step} (of {self.steps})'
            f'\tindex: {self.sample_idx}'
            f'\tpoints_lr: {points_lr:3.3}'
            f'\tsigma_lr: {sigma_lr:3.3}'
            f'\tmin KLDiv: {kldiv.min():3.3}'
            f'\tmin_sigma:{self.mix.sigma.min():3.6}'
            f'\tmax_sigma:{self.mix.sigma.max():3.6}'
            # f'\tpoints:{self.curve.points.flatten()}'
            )
        if self.print_fn:
          self.print_fn(current_mixture)

    return self.curve.points


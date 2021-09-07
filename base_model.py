import torch as t
from torch import nn, optim
import xent

class BaseModel(nn.Module):
  """The Generative model.  Produces the Gaussian Mixture for a rendered
  Bezier Curve"""

  def __init__(self):
    super(BaseModel, self).__init__()
    self.kldivloss = xent.KLDivLoss()
    self.pgmap = { 'mus': 0, 'sigma': 1 }
 
  def init(self):
      self.means.init()

  def cuda(self):
    super(BaseModel, self).cuda()
    param_groups = [
        { 'params': self.means.params() },
        { 'params': self.mix.params()},
        ]

    self.opt = optim.Adam(param_groups, lr=0.0)
    # self.opt = optim.SGD(param_groups, lr=0.0, momentum=0.8)

  def set_lr(self, group, lr):
    self.opt.param_groups[self.pgmap[group]]['lr'] = lr

  def get_lr(self, group):
    return self.opt.param_groups[self.pgmap[group]]['lr']

  def forward(self):
    """Produce the data to be compared to the target"""
    means = self.means()
    mixture = self.mix(means)
    return mixture

  def post_step_hook(self):
    raise NotImplementedError()

  def infer(self, trg_dist, sample_idx, steps, every, print_fn):
    """Does gradient descent on the points in the curve"""
    sched = {}
    sched['mus'] = {0: 1e-2, 30000: 1e-3, 40000: 1e-4, 50000: 2e-6}
    sched['sigma'] = {0: 1e-2, 30000: 5e-4, 40000: 3e-4, 50000: 1e-4 }

    with t.no_grad():
      trg_dist = self.mix.process(trg_dist)
    trg_dist_log = t.where(trg_dist > 0.0, trg_dist.log(), t.zeros_like(trg_dist))
    trg_h = - (trg_dist * trg_dist_log).sum(dim=(1,2))

    for step in range(steps):
      self.opt.zero_grad()
      for k, v in self.pgmap.items():
        if step in sched[k]:
          self.set_lr(k, sched[k][step])

      current_mixture = self() # weird, but true
      l = self.kldivloss(trg_dist, current_mixture)
      l.sum().backward()
      self.opt.step()
      self.post_step_hook()
      """
      """
      # Learning rate scheduling should be applied *after* optimizer's update
      # see https://pytorch.org/docs/stable/optim.html 'How to adjust Learning
      # Rate'

      if step % every == 0:
        # print(f'Step: {step}')
        sigma_lr = self.get_lr('sigma')
        mus_lr = self.get_lr('mus')
        kldiv = l - trg_h 
        print(
            f'Step: {step} (of {steps})'
            f'\tindex: {sample_idx}'
            f'\tmus_lr: {mus_lr:3.3}'
            f'\tsigma_lr: {sigma_lr:3.3}'
            f'\tmin KLDiv: {kldiv.min():3.3}'
            f'\tmin_sigma:{self.mix.sigma.min():3.6}'
            f'\tmax_sigma:{self.mix.sigma.max():3.6}'
            )
        if print_fn:
          print_fn(current_mixture)


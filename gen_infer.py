"""
# Idea: create the generative model, which computes a W x H pixellated grid of
# a Gaussian mixture.  The mixture has T components equally weighted, each a
# Gaussian centered at B(t, P), (with some isotropic covariance) where B is the
# Bezier curve evaluated at time t with anchor points P, with T times equally
# spaced in [0, 1]

# The generative model is first given a randomly initialized P.  Then,
# iteratively it is used to generate this pixellated Gaussian mixture.
# Separately, a target image, which is produced from some unknown Bezier curve,
# is converted into a Gaussian mixture density by:
# 1. Thresholding the pixel values to create a yes/no decision whether a
# mixture component exists there
# 2. Evaluating the mixture at all W x H points
# 3. Normalizing

# Finally, the loss is computed as the KL divergence between the two mixtures,
# and gradient descent on P is performed until convergence
"""
import sys
import torch as t
import numpy as np
from torch import nn, optim
from math import factorial as fac
from PIL import Image
import math
import cv2

# Uses Bernstein Polynomial Form
# b_{v,n}(t) := (n-choose-v) t^v (1-t)^(n-v)

# Let P = [[x0, y0], [x1,y1], ..., [xn,yn]]
# Let n = len(P) - 1 
# Bezier(t,P) := sum_{v \in 0..n} { b_{v,n}(t) P[v] }

class BezierCurve(nn.Module):
  """Calculate the evaluation points of a Bezier"""
  def __init__(self, B, P, nx, ny, T):
    super(BezierCurve, self).__init__()
    self.P = P
    self.T = T
    self.points = t.empty((B, P, 2), requires_grad=True)
    self.nx = nx
    self.ny = ny
    deg = P - 1
    self.coeff = t.tensor([fac(deg) / (fac(v) * fac(deg-v)) for v in range(P)])
    # vs = t.arange(P)
    self.ls = t.empty((B, T), requires_grad=True)
    # self.ls = t.linspace(0, 1, T).unsqueeze(1)

    # bernstein: T,n 
    # self.bernstein = coeff * ls**vs * (1.0-ls)**(deg-vs)

  def init_points(self):
    nn.init.uniform_(self.points, a=0.0, b=1.0)
    with t.no_grad():
      self.points[...,0] *= self.nx
      self.points[...,1] *= self.ny
      self.ls[:] = t.linspace(0, 1, self.T)

  def set_points(self, points):
    with t.no_grad():
      self.points[:] = points

  def forward(self):
    # bernstein: T,P   points: B,P,2 
    # curve: B,T,2
    vs = t.arange(self.P)
    deg = self.P - 1
    self.bernstein = (self.coeff 
            * self.ls.unsqueeze(2) ** vs 
            * (1.0 - self.ls.unsqueeze(2)) ** (deg - vs)
            )
    curve = self.bernstein @ self.points
    return curve


class Mixture(nn.Module):
  """Output a pixellated Gaussian mixture from a curve"""
  def __init__(self, B, nx, ny, sigma):
    super(Mixture, self).__init__()
    self.nx = nx
    self.ny = ny
    self.sigma = t.full((B,), sigma, requires_grad=True)

  def rectify(self, mixture):
    max_vals = 1.0 / (2.0 * self.sigma ** 2)
    rect = t.min(mixture, max_vals.view(-1, 1, 1))
    return rect

  def normalize(self, mixture):
    return mixture / t.sum(mixture, dim=(1,2), keepdim=True)

  def process(self, mixture):
    # mixture = self.rectify(mixture)
    mixture = self.normalize(mixture)
    return mixture

  def init_sigma(self, sigma):
    with t.no_grad():
      self.sigma.fill_(sigma)

  def forward(self, curve):
    """Output the mixture grid from the Bezier points
    curve: B,T,2
    output: B,T,T
    """
    T = curve.shape[1]
    xp = t.linspace(0.5, self.nx - 0.5, self.nx).unsqueeze(0)
    yp = t.linspace(0.5, self.ny - 0.5, self.ny).unsqueeze(0)
    bx = curve[...,0].unsqueeze(2)
    by = curve[...,1].unsqueeze(2)
    rsig = (1.0 / (2.0 * self.sigma ** 2)).view(-1, 1, 1)
    # gx: T,nx   gy: T,ny
    # xp - bx: B,T,nx,   yp - by: B,T,ny
    gx = (- rsig * (xp - bx) ** 2).exp()
    gy = (- rsig * (yp - by) ** 2).exp()
    grid = t.einsum('btx,bty -> bxy', gx, gy)
    grid = self.process(grid)
    return grid

class CrossEntropy(t.autograd.Function):
  @staticmethod
  def forward(ctx, p, q):
    lq = q.log()
    ctx.save_for_backward(p, q, lq)
    terms = t.where(p > 0.0, - p * lq, p) 
    return terms.sum(dim=tuple(range(1, terms.ndim)))

  @staticmethod
  def backward(ctx, grad_output):
    p, q, lq = ctx.saved_tensors
    go = grad_output.view(-1, 1, 1)
    p_grad = t.where(p > 0.0, - lq * go, go)
    q_grad = t.where(p > 0.0, - go * p / q, t.zeros_like(q))
    return p_grad, q_grad


class KLDivLoss(nn.Module):
  def __init__(self):
    super(KLDivLoss, self).__init__()

  def forward(self, p, q):
    return CrossEntropy.apply(p, q) 


class BezierModel(nn.Module):
  """The Generative model.  Produces the Gaussian Mixture for a rendered
  Bezier Curve"""

  def __init__(self, P, B, T, nx, ny, sigma, steps, report_every=1, print_callback=None):
    super(BezierModel, self).__init__()
    self.curve = BezierCurve(B, P, nx, ny, T)
    self.mix = Mixture(B, nx, ny, sigma)
    self.kldivloss = KLDivLoss()
    self.steps = steps
    self.report_every = report_every
    self.print_fn = print_callback
    self.pgmap = { 'points': 0, 'sigma': 1, 'ls': 2 }

  def init_points(self):
    self.curve.init_points()

  def set_points(self, points):
    self.curve.set_points(points)

  def init_sigma(self, sigma):
    self.mix.init_sigma(sigma)

  def cuda(self):
    super(BezierModel, self).cuda()
    param_groups = [
        { 'params': self.mix.sigma },
        { 'params': self.curve.points },
        { 'params': self.curve.ls }
        ]
    self.opt = optim.Adam(param_groups, lr=0.0)
    # self.opt = optim.SGD(param_groups, lr=0.0, momentum=0.8)

  def set_lr(self, group, lr):
    self.opt.param_groups[self.pgmap[group]]['lr'] = lr

  def get_lr(self, group):
    return self.opt.param_groups[self.pgmap[group]]['lr']


  def forward(self):
    """Produce the data to be compared to the target"""
    curve = self.curve()
    mixture = self.mix(curve)
    return mixture

  def infer(self, trg_dist):
    """Does gradient descent on the points in the curve"""
    sched = {}
    sched['points'] = {0: 0.0, 30000: 1e-2, 40000: 1e-4, 50000: 2e-6}
    sched['sigma'] = {0: 5e-3, 30000: 5e-4, 40000: 3e-4, 50000: 1e-4 }
    sched['ls'] = {0: 1e-6}

    # sigma_sched = {0: 1e-2 }
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
            f'Step: {step}'
            f'\tpoints_lr: {points_lr:3.3}'
            f'\tsigma_lr: {sigma_lr:3.3}'
            f'\tKLDiv: {kldiv.min():3.3}'
            f'\tmin_sigma:{self.mix.sigma.min():3.6}'
            f'\tls:{self.curve.ls[0,0:10]}'
            # f'\tpoints:{self.curve.points.flatten()}'
            )
        if self.print_fn:
          self.print_fn(current_mixture)

    return self.curve.points


def print_mixture(mixture, path):
  """Print the mixture to an image file"""
  img = Image.new(mode='L', size=mixture.shape)
  mix_np = mixture.detach().numpy().flatten()
  mix_np *= 256 / np.max(mix_np)
  img.putdata(mix_np)
  img.save(path)


def main():
  t.set_printoptions(linewidth=150, precision=3)
  if len(sys.argv) != 10:
    print('Usage: gen_infer.py <img_dir> <target_file> <idx> '
    '<num_bezier_points> <B> <T> <sigma> <report_every> <steps>')
    raise SystemExit(0)

  img_dir, target_file, idx, num_points, B, T, sigma, report_every, steps = sys.argv[1:]
  print(f'img_dir: {img_dir}\n'
      f'target_file: {target_file}\n'
      f'idx: {idx}\n'
      f'num_bezier_points: {num_points}\n'
      f'B: {B}\n'
      f'T: {T}\n'
      f'sigma: {sigma}\n'
      f'report_every: {report_every}\n'
      f'steps: {steps}\n'
      )

  idx = int(idx)
  num_points = int(num_points)
  T = int(T)
  B = int(B)
  sigma = float(sigma)
  steps = int(steps)
  report_every = int(report_every)

  target_points = np.load(f'{img_dir}/{target_file}')
  img_path = f'{img_dir}/d{idx}.png'

  img = Image.open(img_path).convert(mode='L')
  img_data = t.from_numpy(np.array(img, dtype=np.float32))
  nx, ny = img_data.shape

  fourcc = cv2.VideoWriter_fourcc(*'DIVX')
  vwriter = cv2.VideoWriter(f'{img_dir}/{idx}.avi', fourcc, 20, (nx*B, ny))
  def print_fn(img_data):
    # img_data: B,nx,ny
    d = img_data.detach().numpy()
    d *= 255.0 / np.max(d, axis=(1,2), keepdims=True)
    d = np.concatenate([d[i] for i in range(d.shape[0])], axis=1)
    d = np.tile(np.expand_dims(d, 2), 3).astype(np.uint8)
    vwriter.write(d)

  model = BezierModel(num_points, B, T, nx, ny, sigma, steps, report_every, print_fn)
  # model = BezierModel(num_points, B, T, nx, ny, sigma, steps, report_every, None)
  model.init_points()
  model.cuda()
  img_data.cuda()

  # Experimental
  """
  prev_sigma = model.mix.sigma.item()
  model.set_points(t.tensor([5.0, 5.0]))
  model.init_sigma(0.1)
  with t.no_grad():
    img_data = model()
  model.init_sigma(prev_sigma)
  """

  # print_mixture(img_data, f'{img_dir}/{idx}.sanity.png')
  # print('image bezier points: ', model.curve.points.flatten())

  points = model.infer(img_data.unsqueeze(0).repeat(B, 1, 1))
  vwriter.release()

  print('inferred: ', points)
  print('target: ', target_points[idx].flatten())

  # cv2.destroyAllWindows()


if __name__ == '__main__':
  main()


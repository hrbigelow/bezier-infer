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

# Uses Bernstein Polynomial Form
# b_{v,n}(t) := (n-choose-v) t^v (1-t)^(n-v)

# Let P = [[x0, y0], [x1,y1], ..., [xn,yn]]
# Let n = len(P) - 1 
# Bezier(t,P) := sum_{v \in 0..n} { b_{v,n}(t) P[v] }

class BezierCurve(nn.Module):
  """Calculate the evaluation points of a Bezier"""
  def __init__(self, num_points, nx, ny, T):
    super(BezierCurve, self).__init__()
    self.num_points = num_points
    self.T = T
    self.points = t.empty((self.num_points, 2), requires_grad=True)
    self.nx = nx
    self.ny = ny
    n = num_points
    deg = num_points - 1
    poly_coeff = t.tensor([fac(deg) / (fac(v) * fac(deg-v)) for v in range(n)])
    vs = t.arange(n)
    ls = t.linspace(0, 1, T).unsqueeze(1)

    # bernstein: T,n 
    self.bernstein = poly_coeff * ls**vs * (1.0-ls)**(deg-vs)

  def init_points(self):
    nn.init.uniform_(self.points, a=0.0, b=1.0)
    with t.no_grad():
      self.points[:,0] *= self.nx
      self.points[:,1] *= self.ny

  def forward(self):
    # T,n 
    curve = self.bernstein @ self.points
    return curve


class Mixture(nn.Module):
  """Output a pixellated Gaussian mixture from a curve"""
  def __init__(self, nx, ny, sigma):
    super(Mixture, self).__init__()
    self.nx = nx
    self.ny = ny
    self.sigma = sigma

  def normalize(self, mixture):
    return mixture / t.sum(mixture)

  def forward(self, curve):
    """Output the mixture grid from the Bezier points
    curve: T,2
    """
    T = curve.shape[0]
    xp = t.linspace(0.5, self.nx - 0.5, self.nx).unsqueeze(0)
    yp = t.linspace(0.5, self.ny - 0.5, self.ny).unsqueeze(0)
    bx = curve[:,0].unsqueeze(1)
    by = curve[:,1].unsqueeze(1)
    rsig = 1.0 / (2.0 * self.sigma ** 2)
    # gx: T,nx   gy: T,ny
    gx = (- rsig * (xp - bx) ** 2).exp()
    gy = (- rsig * (yp - by) ** 2).exp()
    grid = t.einsum('ti,tj -> ij', gx, gy)

    # hack normalization. 
    grid = self.normalize(grid)
    return grid


class BezierModel(nn.Module):
  """The Generative model.  Produces the Gaussian Mixture for a rendered
  Bezier Curve"""

  def __init__(self, num_points, T, nx, ny, sigma, steps, report_every=1, print_callback=None):
    super(BezierModel, self).__init__()
    self.curve = BezierCurve(num_points, nx, ny, T)
    self.mix = Mixture(nx, ny, sigma)
    self.loss = nn.KLDivLoss()
    self.steps = steps
    self.report_every = report_every
    self.opt = optim.Adam([self.curve.points], lr=0)
    self.print_fn = print_callback
    # self.opt = optim.SGD([self.curve.points], lr=self.eps)

  def init_points(self):
    self.curve.init_points()

  def forward(self):
    """Produce the data to be compared to the target"""
    curve = self.curve()
    mixture = self.mix(curve)
    return mixture

  def infer(self, target_img_data):
    """Does gradient descent on the points in the curve"""
    self.curve.init_points()
    schedule = {0: 1e-2, 30000: 1e-5, 50000: 2e-6}
    target_img_data = self.mix.normalize(target_img_data)

    for step in range(self.steps):
      if step in schedule:
        lr = schedule[step]
        for g in self.opt.param_groups:
          g['lr'] = lr

      self.opt.zero_grad()
      current_mixture = self() # weird, but true
      l = self.loss(current_mixture.log(), target_img_data)
      l.backward()
      self.opt.step()
      if step % self.report_every == 0:
        print(f'{step}: lr: {lr} {l}\t\tpoints:{self.curve.points.flatten()}')
        if self.print_fn:
          self.print_fn(current_mixture, step)

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
  if len(sys.argv) != 8:
    print('Usage: gen_infer.py <img_dir> <target_file> <idx> '
    '<num_bezier_points> <sigma> <report_every> <steps>')
    raise SystemExit(0)

  img_dir, target_file, idx, num_points, sigma, report_every, steps = sys.argv[1:]
  print(f'img_dir: {img_dir}\n'
      f'target_file: {target_file}\n'
      f'idx: {idx}\n'
      f'num_bezier_points: {num_points}\n'
      f'sigma: {sigma}\n'
      f'report_every: {report_every}\n'
      f'steps: {steps}\n'
      )

  idx = int(idx)
  num_points = int(num_points)
  sigma = float(sigma)
  steps = int(steps)
  report_every = int(report_every)

  def print_fn(img_data, step):
    print_mixture(img_data, f'{img_dir}/{idx}.s{step:05}.png')

  target_points = np.load(f'{img_dir}/{target_file}')
  img_path = f'{img_dir}/d{idx}.png'

  img = Image.open(img_path).convert(mode='L')
  img_data = t.from_numpy(np.array(img, dtype=np.float32))
  nx, ny = img_data.shape
  model = BezierModel(num_points, 100, nx, ny, sigma, steps, report_every,
      print_fn)
  model.cuda()
  img_data.cuda()

  points = model.infer(img_data)
  print('inferred: ', points)
  print('target: ', target_points[idx].flatten())


if __name__ == '__main__':
  main()


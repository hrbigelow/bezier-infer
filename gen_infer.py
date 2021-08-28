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
import torch as t
import numpy as np
from torch import nn, optim
from math import factorial as fac
from PIL import Image
import sys
import matplotlib.pyplot as plt

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

  def set_points(self, points):
    with t.no_grad():
      self.points[:] = points

  def forward(self):
    # T,n 
    curve = self.bernstein @ self.points
    return curve


class PrintCurve(nn.Module):
  def __init__(self, nx, ny):
    super(PrintCurve, self).__init__()
    plt.autoscale(False)
    # plt.style.use('grayscale')
    self.fig = plt.figure(figsize=(1, 1), dpi=nx, frameon=True)
    self.ax = self.fig.add_axes([0, 0, 1, 1])
    self.ax.axis('off')
    self.ax.set_xlim([0, nx])
    self.ax.set_ylim([0, ny])
    self.fig.set_facecolor('black')
    self.line, = self.ax.plot([], [], color='white', linewidth=2.0)

  def set_curve(self, curve_x, curve_y):
    self.line.set_xdata(curve_x)
    self.line.set_ydata(curve_y)

  def render(self):
    plt.draw()

  def print(self, path):
    plt.draw()
    self.fig.savefig(path)

  def get_img_data(self):
    buf, sz = self.fig.canvas.print_to_buffer()
    arr = np.frombuffer(buf, dtype=np.uint8, count=sz[0]*sz[1]*4)
    arr = arr.reshape(sz[0], sz[1], 4)
    arr = arr[:,:,0]
    return t.tensor(arr)

  def forward(self, curve):
    self.set_curve(curve[:,0], curve[:,1])
    self.render()
    return self.get_img_data()


class Mixture(nn.Module):
  """Output a pixellated Gaussian mixture from a curve"""
  def __init__(self, nx, ny, sigma):
    super(Mixture, self).__init__()
    self.nx = nx
    self.ny = ny
    self.sigma = sigma

  def forward(self, bezier):
    """Output the mixture grid from the Bezier points
    bezier: T,2
    """

    T = bezier.shape[0]
    xp = t.linspace(0, 1, self.nx).unsqueeze(0)
    yp = t.linspace(0, 1, self.ny).unsqueeze(0)
    bx = bezier[:,0].unsqueeze(1)
    by = bezier[:,1].unsqueeze(1)
    rsig = 1.0 / (2.0 * self.sigma ** 2)
    # gx: T,nx   gy: T,ny
    gx = (- rsig * (xp - bx) ** 2).exp()
    gy = (- rsig * (yp - by) ** 2).exp()
    grid = t.einsum('ti,tj -> ij', gx, gy)

    # hack normalization. 
    grid /= t.sum(grid)

    return grid


class KLLoss(nn.Module):
  """Calculate a KL Divergence grid-based loss"""
  def __init__(self):
    super(KLLoss, self).__init__()

  def forward(self, grid, target_grid):
    lg = grid.log()
    lt = target_grid.log()
    return t.sum(target_grid * (lt - lg))


class BezierModel(nn.Module):
  """The Generative model.  Produces the Gaussian Mixture for a rendered
  Bezier Curve"""

  def __init__(self, num_points, T, nx, ny, sigma, dark_threshold, steps):
    super(BezierModel, self).__init__()
    self.curve = BezierCurve(num_points, nx, ny, T)
    self.pc = PrintCurve(nx, ny)
    self.mix = Mixture(nx, ny, sigma)
    self.loss = KLLoss()
    self.steps = steps
    self.threshold = dark_threshold
    self.opt = optim.Adam([self.curve.points], lr=0)
    # self.opt = optim.SGD([self.curve.points], lr=self.eps)

  def get_target(self, img_data):
    """Gets the Gaussian Mixture from a rendered Bezier curve image data"""
    dark_pixels = get_dark_pixels(img_data, self.threshold)
    return self.mix(dark_pixels)

  def save_target_image(self, img_data, path):
    trg = Image.new(mode='L', size=img_data.shape)
    mix = self.get_target(img_data)
    mix *= 256 / t.max(mix) 
    trg.putdata(mix.flatten().numpy())
    trg.save(path)

  def forward(self):
    """Produce the data to be compared to the target"""
    curve = self.curve()
    img_data = self.pc(curve)
    dark_pixels = get_dark_pixels(img_data, self.threshold)
    return self.mix(dark_pixels)

  def infer(self, img_data):
    """Does gradient descent on the points in the curve"""

    target_mixture = self.get_target(img_data)
    self.curve.init_points()

    schedule = {0: 1e-4, 30000: 1e-5, 50000: 2e-6}

    pc = PrintCurve(self.mix.nx, self.mix.ny)

    for step in range(self.steps):
      if step in schedule:
        lr = schedule[step]
        for g in self.opt.param_groups:
          g['lr'] = lr

      self.opt.zero_grad()
      current_mixture = self() # weird, but true
      l = self.loss(current_mixture, target_mixture)
      l.backward()
      self.opt.step()
      if step % 100 == 0:
        print(f'{step}: lr: {lr} {l}\t\tpoints:{self.curve.points.flatten()}')

    return self.curve.points

def get_dark_pixels(img, threshold):
  """Return an array of normalized pixel coordinates with
  value above a threshold"""
  # img: nx,ny
  idx = t.nonzero(t.where(img > threshold, img, t.zeros_like(img)) != 0)
  cen = (idx.float() + 0.5) / (t.tensor(img.shape) + 1)
  return cen


def main():
  t.set_printoptions(linewidth=150, precision=3)
  if len(sys.argv) == 1:
    print('Usage: gen_infer.py <img_dir> <target_file> <idx> <num_points> <sigma> <steps>')
    raise SystemExit(0)

  img_dir, target_file, idx, num_points, sigma, steps = sys.argv[1:]
  idx = int(idx)
  num_points = int(num_points)
  sigma = float(sigma)
  steps = int(steps)

  target_points = np.load(f'{img_dir}/{target_file}')
  img_path = f'{img_dir}/d{idx}.png'
  trg_path = f'{img_dir}/d{idx}.trg.png'

  img = Image.open(img_path).convert(mode='L')
  img_data = t.from_numpy(np.array(img, dtype=np.float32))
  img_data.cuda()
  nx, ny = img_data.shape
  model = BezierModel(num_points, 100, nx, ny, sigma, 200, steps)
  model.cuda()
  model.save_target_image(img_data, trg_path)

  points = model.infer(img_data)
  print('inferred: ', points)
  print('target: ', target_points[idx].flatten())


if __name__ == '__main__':
  main()


  



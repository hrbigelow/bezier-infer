import torch as t
from torch import nn

class Mixture(nn.Module):
  """Output a pixellated Gaussian mixture from a curve"""
  def __init__(self, B, T, nx, ny, sigma, is_single_sigma=True):
    super(Mixture, self).__init__()
    self.nx = nx
    self.ny = ny
    if is_single_sigma:
      self._sigma = t.full((B,), sigma, requires_grad=True)
      self.sigma = self._sigma.view(-1, 1)
    else:
      self._sigma = t.full((B, T), sigma, requires_grad=True)
      self.sigma = self._sigma

  def params(self):
    return self._sigma

  def normalize(self, mixture):
    # using t.no_grad() prevents independent points from drifting offscreen.
    # but then, the Q distribution starts to approach uniform
    # with t.no_grad():
    z = t.sum(mixture, dim=(1,2), keepdim=True)
    return mixture / z;

  def process(self, mixture):
    # mixture = self.rectify(mixture)
    mixture += 1e-20
    mixture = self.normalize(mixture)
    return mixture

  def init_sigma(self, sigma):
    with t.no_grad():
      self.sigma.fill_(sigma)

  def forward(self, curve):
    """Output the mixture grid from the Bezier points
    curve: B,T,2
    output: B,nx,ny
    """
    T = curve.shape[1]
    xp = t.linspace(0.5, self.nx - 0.5, self.nx).unsqueeze(0)
    yp = t.linspace(0.5, self.ny - 0.5, self.ny).unsqueeze(0)
    bx = curve[...,0:1]
    by = curve[...,1:2]
    # gx: T,nx   gy: T,ny
    # xp - bx: B,T,nx,   yp - by: B,T,ny
    gx = (- 0.5 * ((xp - bx) / self.sigma.unsqueeze(2)) ** 2).exp()
    gy = (- 0.5 * ((yp - by) / self.sigma.unsqueeze(2)) ** 2).exp()
    grid = t.einsum('btx,bty -> bxy', gx, gy)
    grid = self.process(grid)
    return grid

  def rotated_covariance(self, curve_grad):
    # curve_grad: B,T,2
    B, T = curve_grad.shape[:2]
    norms = t.einsum('btc,btc -> bt', curve_grad, curve_grad).sqrt()
    cg_normed = curve_grad / norms.unsqueeze(2)
    e1 = t.tensor([[1,0,0,1],[0,-1,1,0]]).float()
    e2 = t.tensor([[1,0,0,1],[0,1,-1,0]]).float()
    cg1 = (cg_normed @ e1).reshape(B,T,2,2) 
    cg2 = (cg_normed @ e2).reshape(B,T,2,2)
    d = t.diag(t.tensor([1.0, 0.5])).view(2,2)
    lam = self.sigma.view(B,1,1) * d # B,2,2
    cov = cg1 @ lam.unsqueeze(1) @ cg2
    return cov

  def forward_bck(self, curve, curve_grad):
    """
    curve: B,T,2
    curve_grad: B,T,2
    returns mix: B,
    """
    xp = t.linspace(0.5, self.nx - 0.5, self.nx)
    yp = t.linspace(0.5, self.ny - 0.5, self.ny)
    grid = t.cartesian_prod(xp, yp).view(1, 1, -1, 2) # 1, 1, xp * yp, 2

    mean = curve
    cov = self.rotated_covariance(curve_grad) # B,T,2,2
    tril = t.linalg.cholesky(cov) # B,T,2,2
    half_log_det = tril.diagonal(dim1=-2, dim2=-1).log().sum(-1) # B,T
    diff = grid - mean.unsqueeze(2) # B, T, xp * yp, 2
    diff = diff.permute(0,1,3,2) # B,T,2,xp*yp
    mdist = t.triangular_solve(diff, tril, upper=False)[0].pow(2).sum(-2) # B,T
    log_pdf = -0.5 * (2.0 * math.log(2.0 * math.pi) + mdist) \
        - half_log_det.unsqueeze(2)
    pdf = log_pdf.exp() # B,T
    mix = pdf.sum(1).view(-1, self.nx, self.ny)
    return self.process(mix)


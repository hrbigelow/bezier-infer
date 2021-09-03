import torch as t
from torch import nn

class CrossEntropy(t.autograd.Function):
  @staticmethod
  def forward(ctx, p, q):
    lq = q.log()
    ctx.save_for_backward(p, q)
    terms = t.where(p > 0.0, - p * lq, p) 
    return terms.sum(dim=tuple(range(1, terms.ndim)))

  @staticmethod
  def backward(ctx, grad_output):
    p, q = ctx.saved_tensors
    go = grad_output.view(-1, 1, 1)
    # p_grad = t.where(p > 0.0, - lq * go, go)
    q_grad = t.where(p > 0.0, - go * p / q, t.zeros_like(q))
    return t.zeros_like(p), q_grad


class KLDivLoss(nn.Module):
  def __init__(self):
    super(KLDivLoss, self).__init__()

  def forward(self, p, q):
    return CrossEntropy.apply(p, q) 



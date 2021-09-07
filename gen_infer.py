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
import os
import torch as t
import numpy as np
from PIL import Image
import math
import cv2
import models


def print_mixture(mixture, path):
  """Print the mixture to an image file"""
  img = Image.new(mode='L', size=mixture.shape)
  mix_np = mixture.detach().numpy().flatten()
  mix_np *= 256 / np.max(mix_np)
  # put a border on it

  img.putdata(mix_np)
  img.save(path)


def main():
  t.set_printoptions(linewidth=150, precision=3)
  if len(sys.argv) != 10:
    print('Usage: gen_infer.py <img_dir> <idx> '
    '<P> <B> <T> <sigma> <report_every> <mode> <steps>')
    raise SystemExit(0)

  img_dir, idx, P, B, T, sigma, report_every, mode, steps = sys.argv[1:]
  print(f'img_dir: {img_dir}\n'
      f'idx: {idx}\n'
      f'P: {P}\n'
      f'B: {B}\n'
      f'T: {T}\n'
      f'sigma: {sigma}\n'
      f'report_every: {report_every}\n'
      f'mode: {mode}\n'
      f'steps: {steps}\n'
      )

  rev = os.popen('git rev-parse --short HEAD').read().strip()
  idx = int(idx)
  P = int(P)
  T = int(T)
  B = int(B)
  sigma = float(sigma)
  steps = int(steps)
  report_every = int(report_every)

  model_type, sigma_type = mode.split('-')

  img_path = f'{img_dir}/d{idx}.png'

  trg_img = Image.open(img_path).convert(mode='L')
  trg_img_data = t.from_numpy(np.array(trg_img, dtype=np.float32))
  nx, ny = trg_img_data.shape

  # fourcc = -1
  ext, code = 'mp4', 'mp4v'
  ext, code = 'avi', 'FMP4'
  fourcc = cv2.VideoWriter_fourcc(*code)
  vpath = f'{img_dir}/s{idx}.r{rev}.T{T}.P{P}.S{sigma}.{mode}.{ext}'
  vwriter = cv2.VideoWriter(vpath, fourcc, 10, (nx*B, ny))
  
  def add_border(img_data):
    # img_data: B,nx,ny,3
    d[:,0,:,:] = 255
    d[:,-1,:,:] = 255
    d[:,:,0,:] = 255
    d[:,:,-1,:] = 255


  def print_fn(img_data):
    # img_data: B,nx,ny
    d = img_data.detach().numpy()
    d *= 255.0 / np.max(d, axis=(1,2), keepdims=True)
    z = np.zeros_like(d)
    ti = np.tile(trg_img_data, (d.shape[0], 1, 1))
    d = np.stack([ti, d, z], axis=3) # convert to RGB
    d[...,1] = np.maximum(d[...,1], trg_img_data) # target is red channel
    d = np.concatenate([d[i] for i in range(d.shape[0])], axis=1)
    d = d.astype(np.uint8)
    vwriter.write(d)

  if model_type == 'bezier':
    model = models.Bezier(P, B, T, nx, ny, sigma, sigma_type == 'single')
  elif model_type == 'scatter':
    model = models.MuScatter(B, T, nx, ny, sigma, sigma_type == 'single')
  else:
    raise RuntimeError('invalid mode: ', mode)

  model.init()
  model.cuda()
  trg_img_data.cuda()

  trg = trg_img_data.unsqueeze(0).repeat(B, 1, 1)
  points = model.infer(trg, idx, steps, report_every, print_fn)
  vwriter.release()


if __name__ == '__main__':
  main()


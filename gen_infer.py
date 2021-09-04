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
import bezier_model as bmod 


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
  if len(sys.argv) != 9:
    print('Usage: gen_infer.py <img_dir> <idx> '
    '<P> <B> <T> <sigma> <report_every> <steps>')
    raise SystemExit(0)

  img_dir, idx, P, B, T, sigma, report_every, steps = sys.argv[1:]
  print(f'img_dir: {img_dir}\n'
      f'idx: {idx}\n'
      f'P: {P}\n'
      f'B: {B}\n'
      f'T: {T}\n'
      f'sigma: {sigma}\n'
      f'report_every: {report_every}\n'
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

  img_path = f'{img_dir}/d{idx}.png'

  trg_img = Image.open(img_path).convert(mode='L')
  trg_img_data = t.from_numpy(np.array(trg_img, dtype=np.float32))
  nx, ny = trg_img_data.shape

  # fourcc = -1
  # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  # vwriter = cv2.VideoWriter(f'{img_dir}/{idx}.mp4', fourcc, 20, (nx*B, ny))
  fourcc = cv2.VideoWriter_fourcc(*'FMP4')
  vwriter = cv2.VideoWriter(f'{img_dir}/sam{idx}.rev{rev}.P{P}.S{sigma}.avi', fourcc, 10, (nx*B, ny))
  

  def print_fn(img_data):
    # img_data: B,nx,ny
    d = img_data.detach().numpy()
    d *= 255.0 / np.max(d, axis=(1,2), keepdims=True)
    d[:,0,:] = 255
    d[:,-1,:] = 255
    d[:,:,0] = 255
    d[:,:,-1] = 255
    d = np.tile(np.expand_dims(d, 3), 3) # convert to RGB
    d[...,1] = np.maximum(d[...,1], trg_img_data) # target is red channel
    d = np.concatenate([d[i] for i in range(d.shape[0])], axis=1)
    d = d.astype(np.uint8)
    vwriter.write(d)

  model = bmod.BezierModel(P, B, T, nx, ny, sigma, steps, idx, report_every, print_fn)
  model.init_points()
  model.cuda()
  trg_img_data.cuda()

  points = model.infer(trg_img_data.unsqueeze(0).repeat(B, 1, 1))
  vwriter.release()


if __name__ == '__main__':
  main()


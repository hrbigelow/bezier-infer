import numpy as np
import torch as t
import gen_infer as gi
import matplotlib.pyplot as plt
import sys
import os


if __name__ == '__main__':
  if len(sys.argv) != 8:
    print('Usage: gen_curves.py <out_dir> <num_data> <num_bezier_points> '
        '<x-pixels> <y-pixels> <sigma> <T>')
    raise SystemExit(0)
    
  out_dir, N, P, nx, ny, sigma, T = sys.argv[1:]

  print(
      f'out_dir: {out_dir}\n'
      f'num_data: {N}\n'
      f'P: {P}\n'
      f'nx: {nx}\n'
      f'ny: {ny}\n'
      f'sigma: {sigma}\n'
      f'T: {T}\n'
      )

  N, P, nx, ny, T = int(N), int(P), int(nx), int(ny), int(T) 
  sigma = float(sigma)

  if not os.path.exists(out_dir):
    os.mkdir(out_dir)

  points = t.empty(N, P, 2)
  bmod = gi.BezierModel(P, 10, T, nx, ny, sigma, 0, 0, 0, None)

  i = 0
  while i != N:
    bmod.init_points()
    with t.no_grad():
      mix = bmod()
    for j in range(mix.shape[0]):
      gi.print_mixture(mix[j], f'{out_dir}/d{i}.png')
      points[i] = bmod.curve.points[j]
      i += 1
      if i == N:
        break

  
  html_file = f'{out_dir}/images.html'
  with open(html_file, 'w') as fh:
    print('<html><body>', file=fh)
    print(f'<pre>num_bezier_points: {P}</pre>', file=fh)
    print(f'<pre>sigma: {sigma}</pre>', file=fh)
    for i in range(N):
      print(f'<img style="padding: 5px" src="{out_dir}/d{i}.png"></img>', file=fh)
      # print(f'<br>', file=fh)
    print('</body></html>', file=fh)

  points_file = f'{out_dir}/points.npy'
  np.save(points_file, points.detach().numpy())

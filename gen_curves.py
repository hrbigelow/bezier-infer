import numpy as np
import torch as t
import gen_infer as gi
import matplotlib.pyplot as plt
import sys


if __name__ == '__main__':
  if len(sys.argv) != 8:
    print('Usage: gen_curves.py <out_dir> <num_data> <num_bezier_points> '
        '<x-pixels> <y-pixels> <sigma> <T>')
    raise SystemExit(0)
    
  out_dir, N, P, nx, ny, sigma, T = sys.argv[1:]
  N, P, nx, ny, T = int(N), int(P), int(nx), int(ny), int(T) 
  sigma = float(sigma)

  points = t.empty(N, P, 2)
  bmod = gi.BezierModel(P, T, nx, ny, sigma, 0)

  for i in range(N):
    bmod.init_points()
    points[i] = bmod.curve.points
    mix = bmod()
    gi.print_mixture(mix, f'{out_dir}/d{i}.png')

  
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

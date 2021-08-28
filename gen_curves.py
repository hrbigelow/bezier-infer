import numpy as np
import torch as t
import gen_infer as gi
import matplotlib.pyplot as plt
import sys


if __name__ == '__main__':
  if len(sys.argv) != 7:
    print('Usage: gen_curves.py <out_dir> <num_data> <num_bezier_points> '
        '<x-pixels> <y-pixels> <sigma>')
    raise SystemExit(0)
    
  out_dir, N, P, nx, ny, sigma = sys.argv[1:]
  N = int(N)
  P = int(P)
  nx = int(nx)
  ny = int(ny)
  sigma = float(sigma)

  points = t.empty(N, P, 2)

  pc = gi.PrintCurve(nx, ny)

  T = 100
  dark_threshold = 250 
  curve = gi.BezierCurve(P, nx, ny, T)
  bmod = gi.BezierModel(P, T, nx, ny, sigma, dark_threshold, 0)

  for i in range(N):
    curve.init_points()
    points[i] = curve.points
    xy = curve.forward()
    xy = xy.detach().numpy()
    x, y = xy[:,0], xy[:,1]
    pc.set_curve(x, y)
    pc.print(f'{out_dir}/d{i}.png')
    bmod.save_target_image(pc.get_img_data(), f'{out_dir}/d{i}.trg.png')

  
  html_file = f'{out_dir}/images.html'
  with open(html_file, 'w') as fh:
    print('<html><body>', file=fh)
    print(f'<pre>num_bezier_points: {P}</pre>', file=fh)
    print(f'<pre>sigma: {sigma}</pre>', file=fh)
    for i in range(N):
      print(f'<img style="border: 5px" src="{out_dir}/d{i}.png"></img>', file=fh)
      print(f'<img style="border: 5px" src="{out_dir}/d{i}.trg.png"></img>', file=fh)
      print(f'<br>', file=fh)
    print('</body></html>', file=fh)

  points_file = f'{out_dir}/points.npy'
  np.save(points_file, points.detach().numpy())

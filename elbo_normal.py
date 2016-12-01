import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Arrow
from matplotlib.patches import Patch
import mpl_toolkits.mplot3d.art3d as art3d

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm

from autograd import grad
from autograd.optimizers import adam

def logpx(x, mu=0.0, log_std=1.0, alpha=0.0):
  return norm.logpdf(x, mu, np.exp(log_std)) + \
         norm.logcdf(alpha * x, mu, np.exp(log_std)) + \
         np.log(2)

def px_sufficient_stats(mu=0.0, log_std=1.0, alpha=0.0):
    delta = alpha / np.sqrt(alpha * alpha + 1)
    mean = mu + np.exp(log_std) * delta * np.sqrt(2 / np.pi)
    var = np.exp(log_std * 2) * (1 - 2 * delta * delta / np.pi)
    return mean, np.sqrt(var)
true_mean, true_std = px_sufficient_stats()

def gaussian_entropy(log_std):
  return 0.5 * (1.0 + np.log(2*np.pi)) + np.sum(log_std)

rs = npr.RandomState(0)
def elbo(params, t):
  mean, log_std = params[0], params[1]
  samples = rs.randn(100, 1) * np.exp(log_std) + mean
  L = gaussian_entropy(log_std) + np.mean(logpx(samples))
  return -L

fig = plt.figure(figsize=(22, 8))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
x = np.linspace(-5, 5, 20)
y = np.linspace(-10, 3, 20)
X, Y = np.meshgrid(x, y)
A = np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T
zs = np.apply_along_axis(lambda x: elbo(x, 0), 1, A)
Z = zs.reshape(X.shape)

def callback(params, t, g):
  print("Iteration {0:} " \
        "lower bound {1:.4f}; " \
        "mean {2:.4f} [{3:.4f}]; " \
        "variance {4:.4f}[{5:.4f}]".format(
      t,
      -elbo(params, t),
      params[0],
      true_mean,
      np.exp(params[1]) ,
      true_std))
  ax1.clear()
  ax1.set_xlim([-10,10])
  ax1.set_ylim(bottom=0)
  mu, log_std = params[0], params[1]
  xs = np.linspace(-10, 10, 800)
  ys = norm.pdf(xs, mu, np.exp(log_std))
  ax1.plot(xs, ys, color='#f3c273', linewidth=2.0)

  ys = np.exp(logpx(xs))
  ax1.fill_between(xs, 0, ys, color='#aaaaaa')
  gray_patch = Patch(color='#aaaaaa', label='$p(x)$')
  yellow_patch = Patch(color='#f3c273', label='$q(x)$')
  ax1.legend(handles=[gray_patch, yellow_patch])

  ax2.clear()
  ax2.set_xlabel('Mean')
  ax2.set_ylabel('Variance')
  ax2.set_zlabel('Negative ELBO')
  ax2.set_zlim([-50, 30])
  ax2.plot_surface(X, Y, Z, cmap=cm.coolwarm, shade=True, cstride=1, rstride=1, zorder=1)
  ax2.contour(X, Y, Z, zdir='z', offset=-50, cmap=cm.coolwarm, zorder=0, levels=np.linspace(0, 30, 30))

  a = Arrow(params[0], params[1], -2 * g[0], -2 * g[1], width=0.5, zorder=2)
  ax2.add_patch(a)
  art3d.pathpatch_2d_to_3d(a, z=-50, zdir="z")
  # ax2.plot([params[0], params[0]],
  #          [params[1], params[1]],
  #          [-50, elbo(params, 0)], '--', linewidth=2.0, zorder=5)
  # ax2.scatter(params[0], params[1], elbo(params, 0), marker='o', s=100)
  plt.draw()
  plt.pause(1.0/30.0)

gradient = grad(elbo)

init_mean    = -3 * np.ones(1)
init_log_std = -5 * np.ones(1)
init_var_params = np.concatenate([init_mean, init_log_std])
variational_params = adam(
    gradient,
    init_var_params,
    step_size=0.1,
    num_iters=100,
    callback=callback)

plt.show()

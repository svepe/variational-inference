from math import log, sqrt, pi, exp

# These are used for plotting
import numpy as np
import scipy.stats as stats
import matplotlib.animation as anim
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Arrow
from matplotlib.patches import Patch
from matplotlib.collections import Collection
import mpl_toolkits.mplot3d.art3d as art3d


def norm_logpdf(x, mu, std):
    return -log(sqrt(2 * pi) * std) - 0.5 * ((x - mu) / std)**2


def kl(q_params, p_params):
    mu_q, var_q = q_params[0], exp(q_params[1])**2
    mu_p, var_p = p_params[0], exp(p_params[1])**2
    return 0.5 * (log(var_p / var_q) +
                  (mu_q - mu_p)**2 / var_p**2 + var_q / var_p - 1)


def grad_mu_kl(q_params, p_params):
    mu_q = q_params[0]
    mu_p, var_p = p_params[0], exp(p_params[1])**2
    return (mu_q - mu_p) / var_p


def grad_std_kl(q_params, p_params):
    std_q = exp(q_params[1])
    var_p = exp(p_params[1])**2
    return -1 / std_q + std_q / var_p


def grad_logstd_kl(q_params, p_params):
    std_q = exp(q_params[1])
    return std_q * grad_std_kl(q_params, p_params)


def grad_descent(init_params, p_params, gamma, iters):
    q_params = init_params
    descent_history = [[q_params[0], q_params[1], 0., 0., kl(q_params, p_params)]]
    for i in range(iters):
        dmu = grad_mu_kl(q_params, p_params)
        dlogstd = grad_logstd_kl(q_params, p_params)
        q_params = (q_params[0] - gamma * dmu, q_params[1] - gamma * dlogstd)
        descent_history.append(
            [q_params[0], q_params[1], dmu, dlogstd, kl(q_params, p_params)])
    return descent_history


if __name__ == '__main__':
    p_params = (0.0, 0.0)
    opt_traj = grad_descent((5.0, -4.0), p_params, 0.02, 350)

    # This code visualises the gradient descent
    fig = plt.figure(figsize=(12, 8))
    ax_fit = plt.subplot2grid((5, 5), (1, 0), colspan=1, rowspan=3)
    ax_fit.set_xlim([-10, 10])
    ax_fit.set_ylim([0., 0.6])
    x = np.linspace(-10, 10, 800)
    y = stats.norm.pdf(x, p_params[0], np.exp(p_params[1]))
    ax_fit.fill_between(x, 0, y, color='#aaaaaa')
    fit_artist = ax_fit.plot([], [], color='#f3c273', linewidth=3.0)
    ax_fit.legend(handles=[Patch(color='#aaaaaa', label='$p(x)$'),
                           Patch(color='#f3c273', label='$q(x)$')],
                  framealpha=0.3)

    ax_grad = plt.subplot2grid((5, 5), (0, 1), colspan=4, rowspan=5, projection='3d', azim=-73.0, elev=25.0)
    ax_grad.set_title('Gradient Descent')
    ax_grad.set_xlabel('Mean')
    ax_grad.set_ylabel('log(Variance)')
    ax_grad.set_zlabel('KL Divergence')
    ax_grad.set_zlim([-50, 50])
    ax_grad.set_zticks([0, 20, 40])
    x = np.linspace(-5, 5, 20)
    y = np.linspace(-10, 2, 20)
    X, Y = np.meshgrid(x, y)
    A = np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T
    zs = np.apply_along_axis(lambda params: kl(params, p_params), 1, A)
    Z = zs.reshape(X.shape)
    ax_grad.plot_surface(
        X, Y, Z, cmap=cm.coolwarm, shade=True, cstride=1, rstride=1)
    ax_grad.contour(
        X, Y, Z, zdir='z', offset=-50, cmap=cm.coolwarm, levels=np.linspace(0, 30, 30))

    plt.tight_layout()
    for t in range(len(opt_traj)):
        mu, logstd, dmu, dlogstd, kl_score = opt_traj[t]

        # Plot new fit
        xs = np.linspace(-10, 10, 800)
        ys = stats.norm.pdf(xs, mu, np.exp(logstd))
        ax_fit.set_title("$\mathcal{D}_{KL} = $" + "{0:.4f}".format(kl_score))
        fit_artist[0].set_data(xs, ys)

        # Plot gradient step
        grad_step = Arrow(mu, logstd, -dmu, -dlogstd, width=0.5)
        if len(ax_grad.patches) > 0:
            ax_grad.patches.pop()
        ax_grad.add_patch(grad_step)
        art3d.pathpatch_2d_to_3d(grad_step, z=-50, zdir="z")
        # Save images in order to export a video with:
        # ffmpeg -r 25 -i %04d_kl_normal.png kl_normal.webm
        plt.savefig("figs/{0:04d}_kl_normal.png".format(t))
        plt.pause(0.001)

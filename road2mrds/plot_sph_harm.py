import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# The following import configures Matplotlib for 3D plotting.
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm
import sys

l = int( sys.argv[1] )
m = int( sys.argv[2] )

def plot_Y(l, m):
    Y = sph_harm(abs(m), l, phi, theta)

    if m < 0:
        Y = np.sqrt(2) * (-1)**m * Y.imag
    elif m > 0:
        Y = np.sqrt(2) * (-1)**m * Y.real

    Yx, Yy, Yz = np.abs(Y)*x, np.abs(Y)*y, np.abs(Y)*z

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(Yx, Yy, Yz,
                    rstride=2, cstride=2)
    ax_lim = 0.5
    ax.plot([-ax_lim, ax_lim], [0,0], [0,0], c='0.5', lw=1, zorder=10)
    ax.plot([0,0], [-ax_lim, ax_lim], [0,0], c='0.5', lw=1, zorder=10)
    ax.plot([0,0], [0,0], [-ax_lim, ax_lim], c='0.5', lw=1, zorder=10)
    # Set the Axes limits and title, turn off the Axes frame.
    ax.set_title(r'$Y_{{{},{}}}$'.format(l, m))
    ax_lim = 0.5
    ax.set_xlim(-ax_lim, ax_lim)
    ax.set_ylim(-ax_lim, ax_lim)
    ax.set_zlim(-ax_lim, ax_lim)
    ax.axis('off')

    plt.show()

theta = np.linspace(0, np.pi, 100)
phi   = np.linspace(0, 2*np.pi, 100)
theta, phi = np.meshgrid(theta, phi)
x,y,z = np.sin(theta)*np.sin(phi), np.sin(theta)*np.cos(phi), np.cos(theta)

plot_Y(l, m)


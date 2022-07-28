"""
NOTE:
Ignore this file
It does not work
"""


import numpy as np
import nibabel as nib
import itertools
from random import random
from scipy.special import hyp1f1 as M
from utils import *

# Compute a rotation matrix corresponding to the orientation (azimuth,zenith)
#     azimuth (phi):	angle in the x-y plane
#     zenith  (theta):	angle in the x-z plane
# ---------------------------------------------------------------------------
def get_rotation(azimuth, zenith):
    azimuth = azimuth % (2*np.pi)
    zenith  = zenith % (np.pi)

    azimuth_rotation = np.array([
        [np.cos(azimuth), -np.sin(azimuth), 0.0],
        [np.sin(azimuth),  np.cos(azimuth), 0.0],
        [0.0,              0.0,             1.0]
    ])

    zenith_rotation = np.array([
        [ np.cos(zenith), 0.0, np.sin(zenith)],
        [ 0.0,            1.0,            0.0],
        [-np.sin(zenith), 0.0, np.cos(zenith)]
    ])

    #return np.matmul(zenith_rotation, azimuth_rotation) # primero aplica la zenith
    #return np.matmul(azimuth_rotation, zenith_rotation) # primero aplica la azimuth

    return zenith_rotation @ azimuth_rotation
    #return azimuth_rotation @ zenith_rotation

def get_scale(u, v, kappa):
    return (1/M(1.0/2.0,3.0/2.0,kappa)) * np.exp( kappa*((u@v)**2) )

def get_acquisition(voxel, scheme, sigma=0.0):
    nsamples = len(scheme)
    signal = np.zeros(nsamples)

    for i in range(nsamples):
        signal[i] = E(g=scheme[i][0:3], b=scheme[i][3], voxel=voxel)
        # todo: add noise to the signal

    return signal

def tensor_signal(g, b, pdd, lambdas):
    R = get_rotation(pdd[0], pdd[1])
    D = R @ np.diag(lambdas) @ R.transpose()

    return np.exp( -b*(g.transpose()@D@g) )

# Probe the signal at a given q-space coordinate for a given voxel configuration
# ------------------------------------------------------------------------------
def E(g, b, voxel, dispersion=False, noise=False):
    ntensors = voxel['nfascicles']
    pdds = voxel['pdds']
    lambdas = voxel['eigenvals']
    ndirs = voxel['ndirs']
    kappa = voxel['kappa']
    alphas = voxel['fractions']

    signal = 0
    for i in range(ntensors):
        if dispersion:
            dirs,weights = get_dispersion(pdds[0], pdds[1], ndirs, kappa)

            dispersion_signal = 0
            for j in range(ndirs):
                dispersion_signal = dispersion_signal + weights[j]*tensor_signal(g, b, dirs[j], lambdas[i])

            signal = signal + alphas[i]*dispersion_signal
        else:
            signal = signal + alphas[i]*tensor_signal(g, b, pdds[i], lambdas[i])

    return signal

# Return the directions and the weights of the dispersion
# (azimuth, zenith): spherical coordinates of the PDD
# ndirs: number of directions in the dispersion
# kappa: concentration parameter
# ------------------------------------------------------------------------------
def get_dispersion(azimuth, zenith, ndirs, kappa):
    p = sph2cart(1.0, azimuth, zenith)

    eps_azimuth = random_angles(min_angle=-np.pi/2, max_angle=np.pi/2, size=ndirs)
    eps_zenith  = random_angles(min_angle=-np.pi/2, max_angle=np.pi/2, size=ndirs)

    points,dirs, weights = [],[],[]
    for i in range(ndirs):
        R = get_rotation(azimuth+eps_azimuth[i], zenith+eps_zenith[i])
        #dirs.append( R )
        #a,b = R@p, R@(-p)
        dirs.append( np.array([azimuth+eps_azimuth[i], zenith+eps_zenith[i]]) )
        q = sph2cart(1, azimuth+eps_azimuth[i], zenith+eps_zenith[i])

        s = get_scale(p, q, kappa)
        weights.append( s )
        """a,b = s*a,s*b

        points = points + [a,b]#"""

    return np.array(points), np.array(dirs), np.array(weights)

def plot_dispersion_old(points):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    X,Y,Z = points[:,0], points[:,1], points[:,2]
    ax.scatter(X, Y, Z, color='blue')

    # fake bounding box to fix axis size
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    plt.show()

def plot_dispersion(azimuth, zenith, dirs, weights):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    p = sph2cart(1.0, azimuth, zenith)
    points = []
    for ((azimuth, zenith),weight) in zip(dirs, weights):
        #a,b = R@p, R@(-p)
        #a,b = s*a, s*b
        #points = points + [a,b]
        q = sph2cart(1, azimuth, zenith)
        q = weight*q
        points = points + [q, -q]
    points = np.array(points)

    X,Y,Z = points[:,0], points[:,1], points[:,2]
    ax.scatter(X, Y, Z, color='blue')

    #ax.scatter(p[0], p[1], p[2], color='red', marker='*')
    #ax.scatter(-p[0], -p[1], -p[2], color='red', marker='+')

    #surf = ax.plot_trisurf(X, Y, Z, linewidth=0)

    # fake bounding box to fix axis size
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    ax.plot([-max_range, max_range], [0,0], [0,0], c='0.5', lw=1, zorder=10)
    ax.plot([0,0], [-max_range, max_range], [0,0], c='0.5', lw=1, zorder=10)
    ax.plot([0,0], [0,0], [-max_range, max_range], c='0.5', lw=1, zorder=10)
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    plt.show()

def plot_angles(azimuth, zenith, nangles, kappa):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(20,10))  # Square figure
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(r'$\kappa = %d$' % kappa)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    mu, sigma = 0, 0.05
    max_angle = 2
    #eps_azimuth = np.random.normal(mu, sigma, nangles)
    #eps_zenith  = np.random.normal(mu, sigma, nangles)
    #eps_azimuth = np.random.rand(nangles)*0.01
    #eps_zenith  = np.random.rand(nangles)
    eps_azimuth = random_angles(a=-np.pi/max_angle, b=np.pi/max_angle, size=nangles)
    eps_zenith  = random_angles(a=-np.pi/max_angle, b=np.pi/max_angle, size=nangles)

    line = np.array([
        np.array([0.0, 1.0,  0.0]), #endpoints
        np.array([0.0, -1.0, 0.0])
    ])

    line = line @ get_rotation(azimuth, zenith)

    #ax.plot(line[:,0], line[:,1], line[:,2], marker='o', color='red')

    points = []

    for i in range(nangles):
        R = get_rotation(azimuth+eps_azimuth[i], zenith+eps_zenith[i])
        #R = get_rotation(azimuth+eps_azimuth[i], zenith)
        #R = get_rotation(azimuth, zenith+eps_zenith[i])
        #R = get_rotation(azimuth, zenith)

        rot_line = line @ R #rotate endpoints

        S = get_scale(line, rot_line, kappa)

        rot_line = rot_line @ S #scale endpoints

        points.append( rot_line[0] )
        points.append( rot_line[1] )

        #ax.scatter(rot_line[:,0], rot_line[:,1], rot_line[:,2], marker='o', color='blue')
        #ax.plot(rot_line[:,0], rot_line[:,1], rot_line[:,2], marker='o', color='blue')

    points = np.array( points )
    X,Y,Z = points[:,0], points[:,1], points[:,2]
    ax.scatter(X, Y, Z, color='blue')

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    plt.show()

scheme = load_scheme('Penthera_3T.txt')
X,Y,Z,N = 16,16,5,len(scheme) # dimensions of the phantom
voxels = itertools.product( range(X), range(Y), range(Z) )

SNRs = [30, 12, np.inf]
SNR = np.inf

numcomp = np.random.randint( low=1, high=4, size=(X,Y,Z) ) # number of fascicles per voxel
fracs   = np.zeros( (X,Y,Z, 3) )                           # volume fractions per voxel
eigens  = np.zeros( (X,Y,Z, 9) )                           # eigenvalues per voxel
pdds    = np.zeros( (X,Y,Z, 6) )                           # principal diffusion directions per voxel
dwi     = np.zeros( (X,Y,Z, N) )                           # volume

"""
for (x,y,z) in voxels:
    n = numcomp[x,y,z]

    voxel = {}
    voxel['xyz'] = (x,y,z)
    voxel['nfascicles'] = n
    voxel['fractions'] = random_fractions(n)
    voxel['eigenvals'] = random_eigenvals(n)
    voxel['pdds'] = random_pdds(n)
    voxel['ndirs'] = 5000
    voxel['kappa'] = 16

    dwi[x,y,z, :] = get_acquisition(voxel, scheme)
#"""

#nib.save( nib.Nifti1Image(dwi, np.identity(4)), 'dwi.nii' )
#nib.save( nib.Nifti1Image(numcomp, np.identity(4)), 'numcomp.nii' )

azimuth=0
zenith=0

#azimuth = random_angles(0, 2*np.pi, 1) [0]
#zenith  = random_angles(0,   np.pi, 1) [0]

points, dirs, weights = get_dispersion(azimuth=azimuth, zenith=zenith, ndirs=5000, kappa=0)
#plot_dispersion_old(points)
plot_dispersion(azimuth=azimuth, zenith=zenith, dirs=dirs, weights=weights)
#"""
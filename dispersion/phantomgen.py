import numpy as np
import nibabel as nib
import itertools
from random import random
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

def get_acquisition(voxel, scheme, noise=False, dispersion=False, sigma=0.0):
    nsamples = len(scheme)
    signal = np.zeros(nsamples)

    for i in range(nsamples):
        signal[i] = E(g=scheme[i][0:3], b=scheme[i][3], voxel=voxel, noise=noise, dispersion=dispersion)
        # todo: add noise to the signal

    return signal

def tensor_signal(g, b, pdd, lambdas):
    e = sph2cart(1, pdd[0], pdd[1])
    
    d = np.dot(g, e)

    gDg = lambdas[1]*(1 - d**2) + lambdas[0]*(d**2)

    return np.exp( -b*(gDg) )

# Probe the signal at a given q-space coordinate for a given voxel configuration
# ------------------------------------------------------------------------------
def E(g, b, voxel, noise=False, dispersion=False):
    ntensors = voxel['nfascicles']
    pdds = voxel['pdds']
    R = voxel['rotmats']
    lambdas = voxel['eigenvals']
    ndirs = voxel['ndirs']
    kappa = voxel['kappa']
    alphas = voxel['fractions']
    sigma = voxel['sigma']

    signal = 0
    for i in range(ntensors):
        if dispersion:
            pdd = pdds[i]
            dirs,weights = get_dispersion(pdd[0], pdd[1], ndirs, kappa)

            dispersion_signal = 0
            for j in range(ndirs):
                dispersion_signal += weights[j]*tensor_signal(g, b, dirs[j], lambdas[i])

            signal += alphas[i]*dispersion_signal
        else:
            signal += alphas[i]*tensor_signal(g, b, pdds[i], lambdas[i])

    if noise:
        s = sigma * np.random.normal(loc=0, scale=1, size=2)
        signal = np.sqrt( (signal + s[0])**2 + s[1]**2 )

    return signal

bundle1_rotmat = np.array([
    [-0.1227878 , -0.70710678,  0.69636424],
    [-0.1227878 ,  0.70710678,  0.69636424],
    [-0.98480775,  0.        , -0.17364818]
])

bunndle2_rotmat = np.array([
    [ 0.1227878 ,  0.70710678,  0.69636424],
    [-0.1227878 ,  0.70710678, -0.69636424],
    [-0.98480775,  0.        ,  0.17364818]
])

numcomp_filename = 'Training_3D_SF__SNR=Inf__LEFT_NUMCOMP.nii'
compsize_filename = 'Training_3D_SF__SNR=Inf__LEFT_COMPSIZE.nii'
numcomp = nib.load( numcomp_filename ).get_fdata().astype(np.uint8)
compsize = nib.load( compsize_filename ).get_fdata()

scheme = load_scheme('Penthera_3T.txt')
X,Y,Z = numcomp.shape # dimensions of the phantom
N = len(scheme)
voxels = itertools.product( range(X), range(Y), range(Z) )

SNRs = [30, 12, np.inf]
SNR = np.inf

eigens  = np.zeros( (X,Y,Z, 9) ) # eigenvalues per voxel
pdds    = np.zeros( (X,Y,Z, 6) ) # principal diffusion directions per voxel
dwi     = np.zeros( (X,Y,Z, N) ) # volume

for (x,y,z) in voxels:
    #n = numcomp[x,y,z]
    n = 1

    voxel = {}
    voxel['xyz'] = (x,y,z)
    voxel['nfascicles'] = n
    voxel['fractions'] = compsize[x,y,z]
    voxel['eigenvals'] = [np.array([0.001, 0.0003, 0.0001]), np.array([0.001, 0.0001, 0.0003]), np.array([0.001, 0.0003, 0.0001])]
    voxel['pdds'] = random_pdds(n)
    voxel['rotmats'] = [bundle1_rotmat, bunndle2_rotmat, np.identity(3)]
    voxel['ndirs'] = 50
    voxel['kappa'] = 16
    voxel['sigma'] = 1 / SNRs[1]

    dwi[x,y,z, :] = get_acquisition(voxel, scheme, noise=False, dispersion=False)

nib.save( nib.Nifti1Image(dwi, np.identity(4)), 'dwi.nii' )

"""
azimuth=np.pi/2
zenith=np.pi/4

azimuth = random_angles(0, 2*np.pi, 1) [0]
zenith  = random_angles(0,   np.pi, 1) [0]

points, dirs, weights = get_dispersion(azimuth=azimuth, zenith=zenith, ndirs=5000, kappa=16)
#plot_dispersion_old(points)
plot_dispersion(azimuth=azimuth, zenith=zenith, dirs=dirs, weights=weights)
"""
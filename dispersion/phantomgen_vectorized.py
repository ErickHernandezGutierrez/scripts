import numpy as np
import nibabel as nib
import itertools, sys
from random import random
from utils import *
from scipy.special import hyp1f1 as M

# ---------------------------------------------------------------------------
#     azimuth (phi):	angle in the x-y plane
#     zenith  (theta):	angle in the x-z plane
# ---------------------------------------------------------------------------

"""
pdd: (nvoxels, 3)
ndirs: int
kappa: float
"""
def get_dispersion_vec(pdd, ndirs, kappa):
    nvoxels = pdd.shape[0]

    # (nvoxels, 1)
    azimuth, zenith, r = cart2sph(pdd[:,0], pdd[:,1], pdd[:,2])

    azimuth = np.repeat(azimuth, ndirs).reshape(nvoxels,ndirs)
    zenith  = np.repeat(zenith,  ndirs).reshape(nvoxels,ndirs)

    eps_azimuth = random_angles(min_angle=-np.pi/2, max_angle=np.pi/2, size=(nvoxels,ndirs))
    eps_zenith  = random_angles(min_angle=-np.pi/2, max_angle=np.pi/2, size=(nvoxels,ndirs))

    dirs_azimuth = azimuth + eps_azimuth
    dirs_zenith  = zenith  + eps_zenith

    # dirs (nvoxels, ndirs, 3)
    dirs = sph2cart(np.ones((nvoxels,ndirs)), dirs_azimuth, dirs_zenith)
    print(dirs.shape)

    dotps = np.matmul( dirs, pdd.reshape(nvoxels,3,1) )
    scale = 1/M(1.0/2.0,3.0/2.0,kappa)
    #  weights (nvoxels, ndirs)
    weights = scale * np.exp( kappa*(dotps**2) )

    return dirs, weights

def get_acquisition(pdd, g, b, lambda1, lambda23):
    nvoxels  = pdd.shape[0]
    nsamples = g.shape[0]
    b = np.identity( nsamples ) * b
    
    dot = pdd @ g.transpose()

    ones = np.ones( (nvoxels,nsamples) )

    gDg = lambda23@(ones - dot**2) + lambda1@(dot**2)

    return np.exp( -gDg@b )

def get_acquisition_dispersion(pdd, g, b, lambda1, lambda2, ndirs, kappa):
    G = np.tile(g.transpose(),(nvoxels,1)).reshape(nvoxels, 3, nsamples)

    dirs, weights = get_dispersion_vec(pdd, ndirs, kappa)

    # (nvoxels, ndirs, nsamples)
    DOT = np.matmul(dirs, G)

    UNO = np.ones((nvoxels, ndirs, nsamples))
    LAMBDA1 = UNO * lambda1
    LAMBDA2 = UNO * lambda2

    gDg = LAMBDA2*(UNO - DOT**2) + LAMBDA1*(DOT**2)

    B = np.tile(np.identity(nsamples)*b, (nvoxels,1)).reshape(nvoxels,nsamples,nsamples)

    S = np.exp( -np.matmul(gDg, B) )

    S = S * np.repeat(weights, nsamples).reshape(nvoxels, ndirs, nsamples)

    S = np.sum(S, axis=1)

    return S

def add_rician_noise(S, SNR=np.inf):
    nvoxels  = S.shape[0]
    nsamples = S.shape[1]
    sigma = 1 / SNR

    z = np.random.normal(loc=0, scale=1, size=(nvoxels,nsamples)) * sigma
    w = np.random.normal(loc=0, scale=1, size=(nvoxels,nsamples)) * sigma

    return np.sqrt( (S + z)**2 + w**2 )

output_filename = sys.argv[1]
lambdas_filename = sys.argv[2]

numcomp_filename  = 'data/gt-numcomp.nii'
compsize_filename = 'data/gt-compsize.nii'
mask_filename     = 'data/mask.nii'
pdd_filename      = 'data/gt-pdds.nii'
#lambdas_filename  = 'data/gt-lambdas1.nii'

numcomp  = nib.load( numcomp_filename ).get_fdata().astype(np.uint8)
compsize = nib.load( compsize_filename ).get_fdata()
mask     = nib.load( mask_filename ).get_fdata().astype(np.uint8)
pdds     = nib.load( pdd_filename ).get_fdata()
lambdas  = nib.load( lambdas_filename ).get_fdata()

scheme = load_scheme('data/Penthera_3T.txt')
X,Y,Z = numcomp.shape # dimensions of the phantom
nsamples = len(scheme)
nvoxels = X*Y*Z
voxels = itertools.product( range(X), range(Y), range(Z) )

SNRs = [30, 12, np.inf]
affine = np.identity(4)

"""lambdas = np.array([
    np.array([0.001, 0.0003, 0.0003]), # bundle \
    np.array([0.001, 0.0003, 0.0003]), # bundle /
    np.array([0.001, 0.0003, 0.0003])  # bundle O
])#"""

g = scheme[:,0:3]
b = scheme[:,3]
pdds = pdds.reshape(nvoxels, 9)
lambdas = lambdas.reshape(nvoxels, 6)

dwi = np.zeros((nvoxels,nsamples))
for i in range(0, 3):
    #lambda1  = np.identity( nvoxels ) * lambdas[0,0]
    #lambda23 = np.identity( nvoxels ) * lambdas[0,1]

    lambda1  = np.identity( nvoxels ) * lambdas[:, 2*i]
    lambda23 = np.identity( nvoxels ) * lambdas[:, 2*i+1]
    
    alpha    = np.identity( nvoxels ) * compsize[:,:,:, i].flatten()
    #alpha   = np.identity( nvoxels ) * mask[:,:,:].flatten()
    S = get_acquisition(pdds[:, 3*i:3*i+3], g, b, lambda1, lambda23)
    #S = get_acquisition_dispersion(pdd[:, 3*i:3*i+3], g, b, lambda1[0,0], lambda2[0,0], 1000, 16)
    S = alpha @ S

    dwi += S

#dwi = add_rician_noise(dwi, SNR=SNRs[2])
#mask = np.identity( nvoxels ) * mask[:,:,:].flatten()
#S = mask @ S

dwi = dwi.reshape(X,Y,Z,nsamples)
nib.save( nib.Nifti1Image(dwi, affine), output_filename )

"""
azimuth=np.pi/2
zenith=np.pi/4

azimuth = random_angles(0, 2*np.pi, 1) [0]
zenith  = random_angles(0,   np.pi, 1) [0]

points, dirs, weights = get_dispersion(azimuth=azimuth, zenith=zenith, ndirs=5000, kappa=16)
#plot_dispersion_old(points)
plot_dispersion(azimuth=azimuth, zenith=zenith, dirs=dirs, weights=weights)
"""
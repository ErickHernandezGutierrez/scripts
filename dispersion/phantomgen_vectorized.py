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

# Reconstruct Diffusion Tensor From PDDs and lambdas
# pdds:    (nvoxels, 9) matrix
# lambdas: (nvoxels, 6) matrix
# tensors: [(nvoxels, 6), (nvoxels, 6), (nvoxels, 6)] list of matrices
def get_tensors(pdds, lambdas):
    nvoxels = pdds.shape[0]
    tensors = 3*[np.zeros((nvoxels, 6))]

    for i in range(3):
        for voxel in range(nvoxels):
            R = get_rotation_from_dir( axis=(0,0,1), dir=pdds[voxel, 3*i:3*i+3] )
            L = np.diag([ lambdas[voxel, 2*i], lambdas[voxel, 2*i+1], lambdas[voxel, 2*i+1] ])
            D = R.transpose() @ L @ R
            tensors[i][voxel, 0] = D[0,0]
            tensors[i][voxel, 1] = D[1,1]
            tensors[i][voxel, 2] = D[2,2]
            tensors[i][voxel, 3] = D[0,1]
            tensors[i][voxel, 4] = D[0,2]
            tensors[i][voxel, 5] = D[1,2]

    return tensors

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

#output_filename = sys.argv[1]
#lambdas_filename = sys.argv[2]

subject_path = sys.argv[1]

compsize_filename = subject_path + '/gt/compsize.nii'
lambdas_filename  = subject_path + '/gt/lambdas.nii'
numcomp_filename  = subject_path + '/gt/numcomp.nii'
pdd_filename      = subject_path + '/gt/pdds.nii'
mask_filename     = subject_path + '/mask.nii'

compsize = nib.load( compsize_filename ).get_fdata()
lambdas  = nib.load( lambdas_filename ).get_fdata()
numcomp  = nib.load( numcomp_filename ).get_fdata().astype(np.uint8)
pdds     = nib.load( pdd_filename ).get_fdata()
mask     = nib.load( mask_filename ).get_fdata().astype(np.uint8)

scheme = load_scheme(subject_path + '/Penthera_3T.txt')
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
lambdas = lambdas.reshape(nvoxels, 9)

print('generating dwi for %s' % subject_path)

dwi = np.zeros((nvoxels,nsamples))
for i in range(0, 3):
    #lambda1  = np.identity( nvoxels ) * lambdas[0,0]
    #lambda23 = np.identity( nvoxels ) * lambdas[0,1]

    lambda1  = np.identity( nvoxels ) * lambdas[:, 3*i]
    lambda23 = np.identity( nvoxels ) * lambdas[:, 3*i+1]
    
    alpha    = np.identity( nvoxels ) * compsize[:,:,:, i].flatten()
    #alpha   = np.identity( nvoxels ) * mask[:,:,:].flatten()
    S = get_acquisition(pdds[:, 3*i:3*i+3], g, b, lambda1, lambda23)
    #S = get_acquisition_dispersion(pdd[:, 3*i:3*i+3], g, b, lambda1[0,0], lambda2[0,0], 1000, 16)
    S = alpha @ S

    dwi += S

nib.save( nib.Nifti1Image(dwi.reshape(X,Y,Z,nsamples), affine), subject_path+'/gt/dwi.nii' )

dwi = add_rician_noise(dwi, SNR=12)
#mask = np.identity( nvoxels ) * mask[:,:,:].flatten()
#S = mask @ S

dwi = dwi.reshape(X,Y,Z,nsamples)
nib.save( nib.Nifti1Image(dwi, affine), subject_path + '/dwi-SNR=12.nii' )

"""
# save GT tensors
tensors = get_tensors(pdds, lambdas)
for i in range(3):
    tensors[i] = tensors[i].reshape(X,Y,Z,6)
    nib.save( nib.Nifti1Image(tensors[i], affine), 'gt-tensor-%d.nii' % (i+1) )
"""

"""
azimuth=np.pi/2
zenith=np.pi/4

azimuth = random_angles(0, 2*np.pi, 1) [0]
zenith  = random_angles(0,   np.pi, 1) [0]

points, dirs, weights = get_dispersion(azimuth=azimuth, zenith=zenith, ndirs=5000, kappa=16)
#plot_dispersion_old(points)
plot_dispersion(azimuth=azimuth, zenith=zenith, dirs=dirs, weights=weights)
"""
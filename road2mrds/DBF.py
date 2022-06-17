from logging import exception
import numpy as np
import nibabel as nib
import sys, itertools, math, cmath
from scipy.special import lpmn
from scipy.optimize import nnls
from utils import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#mandar volumen de alphas con las direccciones a mrtrix

DWI_FILENAME    = sys.argv[1]
SCHEME_FILENAME = sys.argv[2]
MASK_FILENAME   = sys.argv[3]
DIRS_FILENAME   = sys.argv[4]
DBF_FILENAME    = sys.argv[5]

dirs = load_directions(DIRS_FILENAME)
ndirs = dirs.shape[0]

dwi  = nib.load(DWI_FILENAME)
mask = nib.load(MASK_FILENAME).get_fdata()
g,b,_ = load_scheme(SCHEME_FILENAME)
S = dwi.get_fdata()

X,Y,Z,N = S.shape
voxels = itertools.product( range(X), range(Y), range(Z) )

design_matrix = np.zeros((N,ndirs), dtype=np.float32)
S_hat = np.zeros((X,Y,Z,N), dtype=np.float32)
alphas = np.zeros((X,Y,Z,ndirs), dtype=np.float32)

lambdas = np.array([0.0003,0.0003,0.015])
for i in range(N):
    for j in range(ndirs):
        Rj = getRotationFromDir( axis=(0,0,1), dir=dirs[j] )
        Tj = Rj.transpose() @ np.diag(lambdas) @ Rj
        design_matrix[i,j] = np.exp( -b[i]*(g[i].transpose()@Tj@g[i]) )

for (x,y,z) in voxels:
    if mask[x,y,z]:
        alphas[x,y,z, :] = nnls(design_matrix, S[x,y,z, :]) [0]
        S_hat[x,y,z, :] = design_matrix@alphas[x,y,z, :]

nib.save( nib.Nifti1Image(alphas , dwi.affine, dwi.header), 'dbf_alphas.nii' )
nib.save( nib.Nifti1Image(S_hat , dwi.affine, dwi.header), DBF_FILENAME )



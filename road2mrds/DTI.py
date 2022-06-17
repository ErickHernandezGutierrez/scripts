import numpy as np
import nibabel as nib
import sys, itertools
from scipy.optimize import nnls
from utils import *

#poner iter 0 y poner -ols en mrtrix para comparar

DWI_FILENAME    = sys.argv[1]
SCHEME_FILENAME = sys.argv[2]
MASK_FILENAME   = sys.argv[3]
DTI_FILENAME    = sys.argv[4]

dwi = nib.load(DWI_FILENAME)
g,b,_ = load_scheme(SCHEME_FILENAME)
S = dwi.get_fdata()
mask = nib.load( MASK_FILENAME ).get_fdata()

X,Y,Z,N = S.shape
voxels = itertools.product( range(X), range(Y), range(Z) ) # voxels=[ (0,0,0), (0,0,1), ..., (X-1,Y-1,Z-1) ]

design_matrix = np.zeros((N,6+1), dtype=np.float32)
dti = np.zeros((X,Y,Z,6), dtype=np.float32)

for i in range(N):
    design_matrix[i, 0] = 1
    design_matrix[i, 1:7] = (-b[i]) * np.array([ g[i,0]**2, g[i,1]**2, g[i,2]**2, 2*g[i,0]*g[i,1], 2*g[i,0]*g[i,2], 2*g[i,1]*g[i,2] ])

for (x,y,z) in voxels:
    if mask[x,y,z]:
        # 2 ways to solve for the tensor
        #tensor = (np.linalg.inv( design_matrix.transpose()@design_matrix ) @ design_matrix.transpose() @ np.log(S[x,y,z, :]))
        tensor = np.linalg.lstsq(design_matrix, np.log(S[x,y,z, :])) [0]
        dti[x,y,z, :] = tensor[1:7]

nib.save( nib.Nifti1Image(dti , dwi.affine, dwi.header), DTI_FILENAME )

import numpy as np
import nibabel as nib
import sys, itertools, math, cmath
from scipy.special import lpmn
from scipy.special import sph_harm
from utils import *

DWI_FILENAME = sys.argv[1]
SCHEME_FILENAME = sys.argv[2]
MASK_FILENAME = sys.argv[3]
SH_FILENAME = sys.argv[4]
lmax = int( sys.argv[5] )
bval = int( sys.argv[6] )

dwi = nib.load(DWI_FILENAME)
mask = nib.load(MASK_FILENAME).get_fdata().astype(np.uint8)
#g,b,idx0 = load_scheme(SCHEME_FILENAME, bval=0)
g,b,idx = load_scheme(SCHEME_FILENAME, bval=bval)
S = dwi.get_fdata()

# get shell
"""g = np.concatenate((g[idx0], g[idx]))
b = np.concatenate((b[idx0], b[idx]))
S = np.concatenate((S[:,:,:,idx0], S[:,:,:,idx]), axis=3) #"""

g = g[idx]
b = b[idx]
S = S[:,:,:,idx]#"""

X,Y,Z,N = S.shape
voxels = itertools.product( range(X), range(Y), range(Z) )

R = int( (lmax+1)*(lmax+2)/2 )
design_matrix = np.zeros((N,R), dtype=np.float32)

SH = np.zeros((X,Y,Z,N), dtype=np.float32)
coefs = np.zeros((X,Y,Z,R), dtype=np.float32)

for i in range(N):
    #print('-------------------------------------------------')
    theta_i, phi_i, r_i = cart2sph(g[i,0], g[i,1], g[i,2])
    #phi_i, theta_i, r_i = cart2sph(g[i,0], g[i,1], g[i,2])

    for l in range(0, lmax+1, 2):
        #print('l=%d: '%l)
        for m in range(-l, l+1):
            #print('\tm=%d'%m)
            j = int( l*(l+1)/2 + m )

            if m < 0:
                Ylm = sph_harm(-m, l, theta_i, phi_i)
                design_matrix[i,j] = np.sqrt(2) * (-1)**m * Ylm.imag
            elif m > 0:
                Ylm = sph_harm(m, l, theta_i, phi_i)
                design_matrix[i,j] = np.sqrt(2) * (-1)**m * Ylm.real
            else:
                design_matrix[i,j] = sph_harm(0, l, theta_i, phi_i).real

for x,y,z in voxels:
    if mask[x,y,z]:
        coefs[x,y,z, :] = np.linalg.inv( design_matrix.transpose()@design_matrix ) @ design_matrix.transpose() @ S[x,y,z, :]
        #coefs[x,y,z, :] = np.linalg.lstsq(design_matrix, S[x,y,z, :]) [0]
        SH[x,y,z, :] = design_matrix @ coefs[x,y,z, :]

nib.save( nib.Nifti1Image(coefs, dwi.affine, dwi.header), '%s_coefs_lmax=%d.nii' % (SH_FILENAME[:-4],lmax) )
nib.save( nib.Nifti1Image(SH, dwi.affine, dwi.header), SH_FILENAME )
#"""
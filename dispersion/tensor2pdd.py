import numpy as np
import nibabel as nib
import itertools
from utils import *
import sys

tensor_filename = sys.argv[1]
mask_filename = sys.argv[2]
pdd_filename = sys.argv[3]

tensor_img = nib.load(tensor_filename)
mask_img   = nib.load(mask_filename)

tensor = tensor_img.get_fdata()
mask   = mask_img.get_fdata().astype(np.uint8)

X,Y,Z = tensor.shape[0:3]
voxels = itertools.product( range(X), range(Y), range(Z) )

pdd = np.zeros( (X,Y,Z,3) )
for (x,y,z) in voxels:
    if mask[x,y,z]:
        e = eigenvectors(tensor[x,y,z, :])

        pdd[x,y,z, :] = e[0]

nib.save( nib.Nifti1Image(pdd, tensor_img.affine), pdd_filename )

"""
azimuth=np.pi/2
zenith=np.pi/4

azimuth = random_angles(0, 2*np.pi, 1) [0]
zenith  = random_angles(0,   np.pi, 1) [0]

points, dirs, weights = get_dispersion(azimuth=azimuth, zenith=zenith, ndirs=5000, kappa=16)
#plot_dispersion_old(points)
plot_dispersion(azimuth=azimuth, zenith=zenith, dirs=dirs, weights=weights)
"""
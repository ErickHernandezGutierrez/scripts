import numpy as np
import nibabel as nib
import itertools, sys
from random import random
from utils import *

left_filename = sys.argv[1]
right_filename = sys.argv[2]
circle_filename = sys.argv[3]

left_pdd_img   = nib.load( left_filename )
right_pdd_img  = nib.load( right_filename )
circle_pdd_img = nib.load( circle_filename )

left_pdd = left_pdd_img.get_fdata()
right_pdd = right_pdd_img.get_fdata()
circle_pdd = circle_pdd_img.get_fdata()

X,Y,Z = left_pdd.shape[0:3]

pdds = np.zeros( (X,Y,Z, 9) )
voxels = itertools.product( range(X), range(Y), range(Z) )

for (x,y,z) in voxels:
    pdds[x,y,z, :] = np.concatenate((left_pdd[x,y,z, :], right_pdd[x,y,z, :], circle_pdd[x,y,z, :]))

nib.save( nib.Nifti1Image(pdds, np.identity(4)), 'gt-pdds.nii' )

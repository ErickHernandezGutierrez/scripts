import numpy as np
import nibabel as nib
import itertools, sys
from utils import *

"""
Remove extra dimension from DIAMOND tensors
reduce from (X,Y,Z,1,6) to (X,Y,Z,6)
"""

TENSOR_FILENAME    = sys.argv[1]

tensor_img = nib.load(TENSOR_FILENAME)

tensor = tensor_img.get_fdata() [:,:,:,0,:]

nib.save( nib.Nifti1Image(tensor, tensor_img.affine, tensor_img.header), TENSOR_FILENAME[:-4]+'_cola.nii' )

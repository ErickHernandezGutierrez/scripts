import numpy as np
import nibabel as nib
import itertools, sys

dti_mrtrix = nib.load( 'tensor.nii' ).get_fdata()
dti_diamond = nib.load( 'calis_diamond/results_t0.nii' ).get_fdata()


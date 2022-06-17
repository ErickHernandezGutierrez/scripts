import numpy as np
import nibabel as nib
import sys, itertools
from utils import save_scheme,load_scheme

"""
Remove bval from dwi and scheme
"""

dwi_filename = sys.argv[1]
scheme_filename = sys.argv[2]
output_scheme_filename = sys.argv[3]
output_dwi_filename = sys.argv[4]
bval = 2000

bvecs, bvals, idx0    = load_scheme(scheme_filename, bval=0)
bvecs, bvals, idx300  = load_scheme(scheme_filename, bval=300)
bvecs, bvals, idx1000 = load_scheme(scheme_filename, bval=1000)
bvecs, bvals, idx2000 = load_scheme(scheme_filename, bval=2000)

print(len(bvecs))
print(len(bvals))
print(len(idx0))
print(len(idx300))
print(len(idx1000))
print(len(idx2000))

idx = idx0 + idx300 + idx1000

bvecs, bvals = bvecs[idx], bvals[idx]

dwi = nib.load(dwi_filename)

output = dwi.get_fdata() [:,:,:,idx]

save_scheme(bvecs, bvals, output_scheme_filename)
nib.save( nib.Nifti1Image(output, dwi.affine, dwi.header), output_dwi_filename )

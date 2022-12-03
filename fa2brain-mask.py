import numpy as np
import nibabel as nib
import sys, itertools

subject = sys.argv[1]

fa_filename = '%s/fa.nii.gz' % subject

fa_file = nib.load( fa_filename )

fa = fa_file.get_fdata()

X,Y,Z = fa.shape
voxels = itertools.product( range(X), range(Y), range(Z) )

brain_mask = np.zeros( (X,Y,Z), dtype=np.uint8 )

for (x,y,z) in voxels:
    if fa[x,y,z] > 0:
        brain_mask[x,y,z] = 1

nib.save( nib.Nifti1Image(brain_mask, fa_file.affine, fa_file.header), '%s/brain-mask.nii' % subject )

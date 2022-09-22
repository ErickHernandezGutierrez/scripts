import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import itertools, sys

fa_filename = sys.argv[1]
output_filename = sys.argv[2]

fa_file = nib.load(fa_filename)
fa = fa_file.get_fdata()

X,Y,Z = fa.shape[0:3]
voxels = itertools.product( range(X), range(Y), range(Z) )

mask = np.zeros( (X,Y,Z), dtype=np.uint8 )

for (x,y,z) in voxels:
    if fa[x,y,z, 0]>0.7 or fa[x,y,z, 1]>0.7 or fa[x,y,z, 2]>0.7:
        mask[x,y,z] = 1

nib.save( nib.Nifti1Image(mask, fa_file.affine, fa_file.header), output_filename )

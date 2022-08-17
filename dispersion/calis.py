import numpy as np
import nibabel as nib
import itertools

compsize = nib.load( 'data/gt-compsize.nii' )

data = compsize.get_fdata()

X,Y,Z = data.shape[0:3]
voxels = itertools.product( range(X), range(Y), range(Z) )

mask = np.zeros((X,Y,Z,3), dtype=np.uint8)

for (x,y,z) in voxels:
    if data[x,y,z, 0] > 0:
        mask[x,y,z, 0] = 1

    if data[x,y,z, 1] > 0:
        mask[x,y,z, 1] = 1

    if data[x,y,z, 2] > 0:
        mask[x,y,z, 2] = 1

nib.save( nib.Nifti1Image(mask[:,:,:,0], compsize.affine), 'mask-left.nii' )
nib.save( nib.Nifti1Image(mask[:,:,:,1], compsize.affine), 'mask-right.nii' )
nib.save( nib.Nifti1Image(mask[:,:,:,2], compsize.affine), 'mask-circle.nii' )

for i in range(X):
    for j in range(Y):
        print('%.1f' % mask[i,j,2,2], end=' ')
    print()
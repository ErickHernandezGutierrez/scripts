import numpy as np
import nibabel as nib
import itertools, sys

DTI_FILENAME = sys.argv[1]
OUTPUT_FILENAME = sys.argv[2]
MODE = sys.argv[3] # mrtrix2diamond or diamond2mrtrix

dti = nib.load(DTI_FILENAME)

xdim,ydim,zdim = dti.shape[0], dti.shape[1], dti.shape[2]
voxels = itertools.product( range(xdim), range(ydim), range(zdim) )

output = np.zeros( (xdim,ydim,zdim,6) )

for (x,y,z) in voxels:
    D = dti.get_fdata()[x,y,z, :]

    if MODE== 'mrtrix2diamond':
        output[x,y,z, 0] = D[0]
        output[x,y,z, 1] = D[3]
        output[x,y,z, 2] = D[4]
        output[x,y,z, 3] = D[1]
        output[x,y,z, 4] = D[5]
        output[x,y,z, 5] = D[2]
    elif MODE=='diamond2mrtrix':
        output[x,y,z, 0] = D[0]
        output[x,y,z, 1] = D[3]
        output[x,y,z, 2] = D[5]
        output[x,y,z, 3] = D[1]
        output[x,y,z, 4] = D[2]
        output[x,y,z, 5] = D[4]

nib.save( nib.Nifti1Image(output, dti.affine, dti.header), OUTPUT_FILENAME )

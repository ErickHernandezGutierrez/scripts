import numpy as np
import nibabel as nib
import itertools, sys

"""
MrTrix Format
0    1    2    3    4    5
D11, D22, D33, D12, D13, D23

Upper Diagonal Format
0    1    2    3    4    5
D11, D12, D13, D22, D23, D33

Lower Diagonal Format
0    1    2    3    4    5
D11, D12, D22, D13, D23, D33
"""

DTI_FILENAME = sys.argv[1]
OUTPUT_FILENAME = sys.argv[2]
MODE = sys.argv[3] #mrtrix2misteri or misteri2mrtrix

print('[INFO] Loading tensor image...')
dti = nib.load(DTI_FILENAME)

X,Y,Z = dti.shape[0], dti.shape[1], dti.shape[2]
voxels = itertools.product( range(X), range(Y), range(Z) )
output = np.zeros( (X,Y,Z,6) )

print('[INFO] Permuting tensor image...')
for (x,y,z) in voxels:
    D = dti.get_fdata()[x,y,z, :]

    if MODE== 'mrtrix2misteri':
        idx = [0, 3, 1, 4, 5, 2] #considering that output will be lower diagonal
    elif MODE=='misteri2mrtrix':
        idx = [0, 2, 5, 1, 3, 4]  #considering that input is lower diagonal
    else:
        print('[ERROR] Option not supported!')

    output[x,y,z, :] = D[idx]

print('[INFO] Saving output image...')

header = dti.header.copy()
if MODE== 'mrtrix2misteri':
    header.set_intent(1005)
    output = output[..., None, :]
elif MODE=='misteri2mrtrix':
    header.set_intent(0)

nib.save( nib.Nifti1Image(output, dti.affine, header), OUTPUT_FILENAME )

print('Done!')
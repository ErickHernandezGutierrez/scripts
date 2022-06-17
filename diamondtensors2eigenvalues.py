from email import header
import numpy as np
import nibabel as nib
import sys, itertools

tensor1_filename = sys.argv[1]
tensor2_filename = sys.argv[2]
tensor3_filename = sys.argv[3]
output_filename  = sys.argv[4]

tensorimgs = []
for filename in [tensor1_filename, tensor2_filename, tensor3_filename]:
    img = nib.load(filename) #load nifti image
    tensorimgs.append( img.get_fdata() )

X,Y,Z = tensorimgs[0].shape[0], tensorimgs[0].shape[1], tensorimgs[0].shape[2]
voxels = itertools.product( range(X), range(Y), range(Z) )

eigenvalues = np.zeros( (X,Y,Z,9), dtype=np.float64 )

for (x,y,z) in voxels:
    for i,tensorimg in enumerate(tensorimgs):
        """
                    0    1    2    3    4    5
        tensor[] = {D11, D12, D13, D22, D23, D33]}
        """
        tensor = tensorimg[x,y,z, 0, :]
        D = np.array([
            np.array([tensor[0], tensor[1], tensor[2]]),
            np.array([tensor[1], tensor[3], tensor[4]]),
            np.array([tensor[2], tensor[4], tensor[5]])
        ])
        lambdas, _ = np.linalg.eig(D)

        eigenvalues[x,y,z, 3*i:3*i+3] = lambdas

nib.save( nib.Nifti1Image(eigenvalues, img.affine, img.header), output_filename )

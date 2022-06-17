from math import ceil
import numpy as np
import nibabel as nib
import sys, itertools

nufo_map_filename = sys.argv[1]
output_filename   = sys.argv[2]
if len(sys.argv) > 3:
    threadshold = float(sys.argv[3])

print('[INFO] Loading input files...')
nufo_map = nib.load( nufo_map_filename )

nufo_data = nufo_map.get_fdata()

X,Y,Z = nufo_data.shape
voxels = itertools.product( range(X), range(Y), range(Z) )

nufo_int = np.zeros( (X,Y,Z), dtype=np.uint8 )

for (x,y,z) in voxels:
    nufo_int[x,y,z] = int(nufo_data[x,y,z] + threadshold)
    #nufo_int[x,y,z] = np.ceil(nufo_data[x,y,z])

nib.save( nib.Nifti1Image(nufo_int, nufo_map.affine, nufo_map.header), output_filename) 

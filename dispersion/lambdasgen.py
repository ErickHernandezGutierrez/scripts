import numpy as np
import nibabel as nib
import sys, itertools

output_filename = sys.argv[1]

#phan-001
lambdas = np.array([
    np.array([0.001, 0.0001]), # lambdas for bundle \ (FA=0.89)
    np.array([0.001, 0.0001]), # lambdas for bundle / (FA=0.89)
    np.array([0.001, 0.0001])  # lambdas for bundle O (FA=0.89)
])

lambdas = np.array([
    np.array([0.001, 0.0001]), # lambdas for bundle \ (FA=0.89)
    np.array([0.001, 0.0001]), # lambdas for bundle / (FA=0.89)
    np.array([0.001, 0.0001])  # lambdas for bundle O (FA=0.89)
])

# not used yet
damaged_mask = np.zeros((16,16,5))
damaged_lambdas = np.array([0.0, 0.0]) # (FA=0.0)
#"""

"""#phan-002
lambdas = np.array([
    np.array([0.001, 0.0001]), # lambdas for bundle \ (FA=0.89)
    np.array([0.001, 0.0001]), # lambdas for bundle / (FA=0.89)
    np.array([0.001, 0.0001])  # lambdas for bundle O (FA=0.89)
])
#"""

"""#phan-003
lambdas = np.array([
    np.array([0.001, 0.0001]), # lambdas for bundle \ (FA=0.89)
    np.array([0.001, 0.0001]), # lambdas for bundle / (FA=0.89)
    np.array([0.002, 0.0009])  # lambdas for bundle O (FA=0.46)
])
#"""

"""#phan-004
lambdas = np.array([
    np.array([0.0030, 0.0001]), # lambdas for bundle \ (FA=0.96)
    np.array([0.0015, 0.0002]), # lambdas for bundle / (FA=0.85)
    np.array([0.0009, 0.0003])  # lambdas for bundle O (FA=0.60)
])
damaged_lambdas = np.array([0.0009, 0.0005]) # (FA=0.34)
damaged_mask = nib.load( 'damage-mask-4.nii' ).get_fdata()
#"""

X,Y,Z = 16,16,5
voxels = itertools.product( range(X), range(Y), range(Z) )

data = np.ones((16,16,5,6))

for (x,y,z) in voxels:
    # for bundle \
    if damaged_mask[x,y,z] == 0:
        data[x,y,z, 0] = lambdas[0, 0]
        data[x,y,z, 1] = lambdas[0, 1]
    else:
        data[x,y,z, 0] = damaged_lambdas[0]
        data[x,y,z, 1] = damaged_lambdas[1]

    # for bundle /
    data[x,y,z, 2] = lambdas[1, 0]
    data[x,y,z, 3] = lambdas[1, 1]

    # for bundle O
    data[x,y,z, 4] = lambdas[2, 0]
    data[x,y,z, 5] = lambdas[2, 1]

nib.save( nib.Nifti1Image(data, np.identity(4)), output_filename )

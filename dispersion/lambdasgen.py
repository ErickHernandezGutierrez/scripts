import numpy as np
import nibabel as nib
import sys, itertools
import matplotlib.pyplot as plt

def L1FA2L2MD(L1, FA):
    a = FA / np.sqrt(3.0 - 2.0*FA*FA)

    MD = L1 / (1.0 + 2.0*a)
    L2 = MD * (1.0 - a)

    return L2

#output_filename = sys.argv[1]

"""
l1 = 1.7 +- .4 mas parecido a humanos, pero peor para MRDS
fa = 0.85 +- 0.1
"""

FA = np.random.normal(loc=0.85, scale=0.03, size=100000)
L1 = np.random.normal(loc=1.7, scale=0.1, size=100000)
L2 = L1FA2L2MD(L1, FA)
fig, ax = plt.subplots(1,3)

ax[0].hist(FA, bins=64, edgecolor='black')
ax[0].set_title('Distribution of FA values')
ax[0].set_xlabel('FA')
ax[0].set_xlim([0.7, 1])
ax[0].grid(True)

ax[1].hist(L1*1e-3, bins=64, edgecolor='black')
ax[1].set_title(r'Distribution of $\lambda_{1}$ values')
ax[1].set_xlabel(r'$\lambda_{1}$')
ax[1].grid(True)

ax[2].hist(L2*1e-3, bins=64, edgecolor='black')
ax[2].set_title(r'Distribution of $\lambda_{2,3}$ values')
ax[2].set_xlabel(r'$\lambda_{2,3}$')
ax[2].grid(True)
plt.show()

#phan-001
lambdas = np.array([
    np.array([0.001, 0.0001]), # lambdas for bundle \ (FA=0.89)
    np.array([0.001, 0.0001]), # lambdas for bundle / (FA=0.89)
    np.array([0.001, 0.0001])  # lambdas for bundle O (FA=0.89)
])
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

nsubjects = 26
nbundles  = 3
X,Y,Z = 16,16,5
voxels = itertools.product( range(X), range(Y), range(Z) )

mask = np.zeros((X,Y,Z,3), dtype=np.uint8)
for i in range(3):
    mask[:,:,:, i] = nib.load('mask-%d.nii' % (i+1)).get_fdata()

L1 = np.random.normal(loc=1.7,  scale=0.1,  size=3*nsubjects)
FA = np.random.normal(loc=0.85, scale=0.03, size=3*nsubjects)

"""
function [L2,MD] = L1FA2L2MD(l1,fa)
% only works for zeppelin model l2 = l3
   a = fa./sqrt(3.0-2.0*fa.*fa);
   MD = l1./(1.0 + 2.0*a);
   L2 = MD.*(1.0 - a);
end
"""

L2 = L1FA2L2MD(L1, FA)

L1 = L1 * 1e-3
L2 = L2 * 1e-3

L1 = np.repeat(L1, X*Y*Z)
L2 = np.repeat(L2, X*Y*Z)

L1 = L1.reshape(nsubjects,3, X,Y,Z)
L2 = L2.reshape(nsubjects,3, X,Y,Z)

for sub_id in range(nsubjects):
    print('generating lambdas for sub-%.3d_ses-1'%(sub_id+1))

    lambdas = np.zeros((X,Y,Z, 9))

    # for bundles \, / and O respectively 
    for bundle_id in range(nbundles):
        lambdas[:,:,:, 3*bundle_id]   = L1[sub_id, bundle_id, :,:,:]
        lambdas[:,:,:, 3*bundle_id+1] = L2[sub_id, bundle_id, :,:,:]
        lambdas[:,:,:, 3*bundle_id+2] = L2[sub_id, bundle_id, :,:,:]

        lambdas[:,:,:, 3*bundle_id]   *= mask[:,:,:, bundle_id]
        lambdas[:,:,:, 3*bundle_id+1] *= mask[:,:,:, bundle_id]
        lambdas[:,:,:, 3*bundle_id+2] *= mask[:,:,:, bundle_id]

    """# bundle \
    lambdas[:,:,:, 0] = L1[:,:,:, 0, sub_id]
    lambdas[:,:,:, 1] = L2[:,:,:, 0, sub_id]
    lambdas[:,:,:, 0] *= mask[:,:,:, 0]
    lambdas[:,:,:, 1] *= mask[:,:,:, 0]

    # bundle /
    lambdas[:,:,:, 2] = L1[:,:,:, 1, sub_id]
    lambdas[:,:,:, 3] = L2[:,:,:, 1, sub_id]
    lambdas[:,:,:, 2] *= mask[:,:,:, 1]
    lambdas[:,:,:, 3] *= mask[:,:,:, 1]

    # bundle O
    lambdas[:,:,:, 4] = L1[:,:,:, 2, sub_id]
    lambdas[:,:,:, 5] = L2[:,:,:, 2, sub_id]
    lambdas[:,:,:, 4] *= mask[:,:,:, 2]
    lambdas[:,:,:, 5] *= mask[:,:,:, 2]"""

    #nib.save( nib.Nifti1Image(lambdas, np.identity(4)), 'sub-%.3d_ses-1/gt/lambdas.nii'%(sub_id+1) ) 

"""
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
#"""

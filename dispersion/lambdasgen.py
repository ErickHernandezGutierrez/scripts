import numpy as np
import nibabel as nib
import sys, itertools
import matplotlib.pyplot as plt

def L1FA2L2MD(L1, FA):
    a = FA / np.sqrt(3.0 - 2.0*FA*FA)

    MD = L1 / (1.0 + 2.0*a)
    L2 = MD * (1.0 - a)

    return L2

def L1L2_TO_FA(L1, L2):
    a = np.sqrt(0.5)
    b = np.sqrt( (L1-L2)**2 + (L2-L2)**2 + (L2-L1)**2 )
    c = np.sqrt( L1**2 + L2**2 + L2**2 )

    return a*b/c

"""FA = np.random.normal(loc=0.85, scale=0.03, size=100000)
L1 = np.random.normal(loc=1.7, scale=0.1, size=100000)
L2 = L1FA2L2MD(L1, FA) #"""

L1 = np.random.normal(loc=1.38e-3, scale=np.sqrt(3.37e-8), size=3*26)
L2 = np.random.normal(loc=2.80e-4, scale=np.sqrt(3.02e-9), size=3*26)
FA = L1L2_TO_FA(L1, L2)

fig, ax = plt.subplots(1,3)

ax[0].hist(FA, bins=64, edgecolor='black')
ax[0].set_title('Distribution of FA values')
ax[0].set_xlabel('FA')
ax[0].set_xlim([0.0, 1])
ax[0].grid(True)

ax[1].hist(L1, bins=64, edgecolor='black')
ax[1].set_title(r'Distribution of $\lambda_{1}$ values')
ax[1].set_xlabel(r'$\lambda_{1}$')
ax[1].grid(True)

ax[2].hist(L2, bins=64, edgecolor='black')
ax[2].set_title(r'Distribution of $\lambda_{2,3}$ values')
ax[2].set_xlabel(r'$\lambda_{2,3}$')
ax[2].grid(True)
plt.show()

"""#phan-001
lambdas = np.array([
    np.array([0.001, 0.0001]), # lambdas for bundle \ (FA=0.89)
    np.array([0.001, 0.0001]), # lambdas for bundle / (FA=0.89)
    np.array([0.001, 0.0001])  # lambdas for bundle O (FA=0.89)
])
damaged_mask = np.zeros((16,16,5))
damaged_lambdas = np.array([0.0, 0.0]) # (FA=0.0)
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

    #nib.save( nib.Nifti1Image(lambdas, np.identity(4)), 'sub-%.3d_ses-1/gt/lambdas.nii'%(sub_id+1) ) 

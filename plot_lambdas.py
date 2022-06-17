import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import sys

EIGENVALUES_FILENAME    = sys.argv[1]

eigenvalues = nib.load(EIGENVALUES_FILENAME).get_fdata()

tensors = []
for i in range(3):
    lambdas1 = eigenvalues[:,:,:, 3*i].flatten() 
    lambdas2 = eigenvalues[:,:,:, 3*i+1].flatten()
    lambdas3 = eigenvalues[:,:,:, 3*i+2].flatten() 
    tensors.append( [lambdas1, lambdas2, lambdas3] )

fig,axs = plt.subplots(3,3)
fig.suptitle('Histograms of lambdas for each tensor - DIAMOND')

for i in range(3):
    for j in range(3):
        axs[i,j].set_title(r'$\lambda_%d$' % (j+1))
        axs[i,j].set_ylabel('Tensor %d' % (i+1))
        axs[i,j].hist(tensors[i][j], bins=64, edgecolor='black')

plt.show()

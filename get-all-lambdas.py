import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import itertools

from utils import tensor2fa

def lambdasFromFile(lambdas_filename):
    with open(lambdas_filename, 'rt') as lambdas_file:
        line = lambdas_file.readlines()[0]
        lambdas = line.split(' ') [0:3]
        lambdas = [float(value) for value in lambdas]

               #lambda1    #lambda2    #lambda3 
        return lambdas[0], lambdas[1], lambdas[2]

def lambdasFromTensor(tensor):
    D = np.zeros((3,3))
    D[0,0] = tensor[0]
    D[1,1] = tensor[1]
    D[2,2] = tensor[2]
    D[0,1] = D[1,0] = tensor[3]
    D[0,2] = D[2,0] = tensor[4]
    D[1,2] = D[2,1] = tensor[5]

    eigenvals = np.linalg.eig(D) [0]
    eigenvals = np.sort(eigenvals)

           #lambda1      #lambda2      #lambda3 
    return eigenvals[2], eigenvals[1], eigenvals[0]

subject_ids =np.arange(start=3, stop=27, step=1)

FA, L1, L2, L3, L23 = [[],[]], [[],[]], [[],[]], [[],[]], [[],[]]

X,Y,Z = 128,170,130
voxels = itertools.product( range(X), range(Y), range(Z) )

for sub_id in subject_ids:
    subject_base_path = './sub-%.3d_ses-1' % sub_id
    print(subject_base_path)

    mask = nib.load('%s/dti-mask.nii' % subject_base_path).get_fdata().astype(np.uint8)

    dti_tensor = nib.load('%s/mrtrix/tensor.nii'%subject_base_path).get_fdata()
    mrtrix_tensor = nib.load('%s/dti/results_DTInolin_Tensor.nii'%subject_base_path).get_fdata()

    for (x,y,z) in voxels:
        if mask[x,y,z]:
            fa = tensor2fa(dti_tensor[x,y,z, :])
            lambda1, lambda2, lambda3 = lambdasFromTensor(dti_tensor[x,y,z, :])
            
            FA[0].append( fa )
            L1[0].append( lambda1 )
            L2[0].append( lambda2 )
            L3[0].append( lambda3 )
            L23[0].append( (lambda2+lambda3)/2.0 )

            fa = tensor2fa(mrtrix_tensor[x,y,z, :])
            lambda1, lambda2, lambda3 = lambdasFromTensor(mrtrix_tensor[x,y,z, :])

            FA[1].append( fa )
            L1[1].append( lambda1 )
            L2[1].append( lambda2 )
            L3[1].append( lambda3 )
            L23[1].append( (lambda2+lambda3)/2.0 )

l1min = max( np.min(L1[0]), np.min(L1[1]) )
l2min = max( np.min(L2[0]), np.min(L2[1]) )
l3min = max( np.min(L3[0]), np.min(L3[1]) )
l23min = max( np.min(L23[0]), np.min(L23[1]) )

l1max = max( np.max(L1[0]), np.max(L1[1]) )
l2max = max( np.max(L2[0]), np.max(L2[1]) )
l3max = max( np.max(L3[0]), np.max(L3[1]) )
l23max = max( np.max(L23[0]), np.max(L23[1]) )

fig, ax = plt.subplots(2,5)

for i in range(2):
    mu    = [np.mean(FA[i]), np.mean(L1[i]), np.mean(L2[i]), np.mean(L3[i]), np.mean(L23[i])]
    sigma = [np.var(FA[i]),  np.var(L1[i]),  np.var(L2[i]),  np.var(L3[i]),  np.var(L23[i])]

    ax[i,0].hist(FA[i], bins=64, edgecolor='black', label=r'hist $FA$')
    ax[i,0].axvline(x=mu[0], color='red', label=r'$\mu$ = %s'%'{:.2e}'.format(mu[0]))
    ax[i,0].axvline(x=sigma[0], color='green', label=r'$\sigma^2$ = %s'%'{:.2e}'.format(sigma[0]), alpha=0.0)
    if i==0:
        ax[i,0].set_title(r'$FA$ values (Rick DTI)')
    else:
        ax[i,0].set_title(r'$FA$ values (MRtrix DTI)')
    ax[i,0].set_xlabel(r'$FA$')
    ax[i,0].set_xlim([0.7,1])
    ax[i,0].grid(True)
    ax[i,0].legend(loc='upper right')

    ax[i,1].hist(L1[i], bins=64, edgecolor='black', color='blue', label=r'hist $\lambda_{1}$')
    ax[i,1].axvline(x=mu[1], color='red', label=r'$\mu$ = %s'%'{:.2e}'.format(mu[1]))
    ax[i,1].axvline(x=sigma[1], color='green', label=r'$\sigma^2$ = %s'%'{:.2e}'.format(sigma[1]), alpha=0.0)
    if i==0:
        ax[i,1].set_title(r'$\lambda_{1}$ values (Rick DTI)')
    else:
        ax[i,1].set_title(r'$\lambda_{1}$ values (MRtrix DTI)')
    ax[i,1].set_xlabel(r'$\lambda_{1}$')
    ax[i,1].set_xlim([l1min, l1max])
    ax[i,1].grid(True)
    ax[i,1].legend(loc='upper right')

    ax[i,2].hist(L2[i], bins=64, edgecolor='black', color='blue', label=r'hist $\lambda_{2}$')
    ax[i,2].axvline(x=mu[2], color='red', label=r'$\mu$ = %s'%'{:.2e}'.format(mu[2]))
    ax[i,2].axvline(x=sigma[2], color='green', label=r'$\sigma^2$ = %s'%'{:.2e}'.format(sigma[2]), alpha=0.0)
    if i==0:
        ax[i,2].set_title(r'$\lambda_{2}$ values (Rick DTI)')
    else:
        ax[i,2].set_title(r'$\lambda_{2}$ values (MRtrix DTI)')
    ax[i,2].set_xlabel(r'$\lambda_{2}$')
    ax[i,2].set_xlim([l2min, l2max])
    ax[i,2].grid(True)
    ax[i,2].legend(loc='upper right')

    ax[i,3].hist(L3[i], bins=64, edgecolor='black', color='blue', label=r'hist $\lambda_{3}$')
    ax[i,3].axvline(x=mu[3], color='red', label=r'$\mu$ = %s'%'{:.2e}'.format(mu[3]))
    ax[i,3].axvline(x=sigma[3], color='green', label=r'$\sigma^2$ = %s'%'{:.2e}'.format(sigma[3]), alpha=0.0)
    if i==0:
        ax[i,3].set_title(r'$\lambda_{3}$ values (Rick DTI)')
    else:
        ax[i,3].set_title(r'$\lambda_{3}$ values (MRtrix DTI)')
    ax[i,3].set_xlabel(r'$\lambda_{3}$')
    ax[i,3].set_xlim([l3min, l3max])
    ax[i,3].grid(True)
    ax[i,3].legend(loc='upper right')

    ax[i,4].hist(L23[i], bins=64, edgecolor='black', color='blue', label=r'hist $\lambda_{2,3}$')
    ax[i,4].axvline(x=mu[4], color='red', label=r'$\mu$ = %s'%'{:.2e}'.format(mu[4]))
    ax[i,4].axvline(x=sigma[4], color='green', label=r'$\sigma^2$ = %s'%'{:.2e}'.format(sigma[4]), alpha=0.0)
    if i==0:
        ax[i,4].set_title(r'$\lambda_{2,3}$ values (Rick DTI)')
    else:
        ax[i,4].set_title(r'$\lambda_{2,3}$ values (MRtrix DTI)')
    ax[i,4].set_xlabel(r'$\lambda_{2,3}$')
    ax[i,4].set_xlim([l23min, l23max])
    ax[i,4].grid(True)
    ax[i,4].legend(loc='upper right')

plt.show()

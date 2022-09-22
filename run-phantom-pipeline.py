import numpy as np
import nibabel as nib
import os, sys, itertools

"""
[root]
├── S1
|   ├── dwi.nii
|   ├── mask.nii (mw mask)
|   └── scheme.txt
└── S2
    ├── *lesion_mask.nii.gz (optional)
    ├── *fodf.nii.gz (optional to generate fixel AFD)
    ├── metrics
    │   └── *.nii.gz
    └── bundles
        └── *.trk
"""

def read_lambdas(lambdas_filename):
    with open(lambdas_filename, 'rt') as lambdas_file:
        line = lambdas_file.readlines()[0]
        lambdas = line.split(' ') [0:3]
        lambdas = [float(value) for value in lambdas]

               #lambda1    #lambda23 
        return lambdas[0], (lambdas[1]+lambdas[2])/2.0

os.system('')

input_path = sys.argv[1]

subjects = ['sub-%d_ses-1'%sub_id for sub_id in range(1,27)]

for subject in subjects:

    # remove noise
    os.system('dwidenoise %s/dwi.nii %s/dwi-denoised.nii -noise %s/sigmas.nii' % (subject,subject,subject))

    # remove Rician bias
    os.system('mrcalc %s/dwi-denoised.nii 2 -pow %s/sigmas.nii 2 -pow -sub 0.001 -max -sqrt %s/dwi-corrected.nii' % (subject,subject,subject))

    # fit DTI
    os.system('mkdir %s/dti' % subject)
    os.system('dti %s/dwi-corrected.nii %s/scheme.txt %s/dti/results.nii -mask %s/mask.nii -response 0 -correction 0 -fa -md' % (subject,subject,subject,subject))

    # get lambdas from DTI
    lambda1, lambda23 = read_lambdas('%s/dti/results_DTInolin_ResponseAnisotropic.txt' % subject)

    # fit MRDS
    os.system('mkdir %s/mrds' % subject)
    os.system('mdtmrds %s/dwi-corrected.nii %s/scheme.txt %s/mrds/results.nii -correction 0 -response %.9f,%.9f -mask %s/mask.nii -modsel bic -each -intermediate -fa -md -mse -method diff 1' % (subject,subject,subject,lambda1,lambda23,subject))

    
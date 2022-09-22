#!/usr/bin/python
import os

def read_lambdas(lambdas_filename):
    with open(lambdas_filename, 'rt') as lambdas_file:
        line = lambdas_file.readlines()[0]
        lambdas = line.split(' ') [0:3]
        lambdas = [float(value) for value in lambdas]

               #lambda1    #lambda23 
        return lambdas[0], (lambdas[1]+lambdas[2])/2.0

for sub_id in range(1, 27):
    subject_base_path = './sub-%.3d_ses-1' % sub_id

    print('---------------------------------------------- %s ----------------------------------------------' % subject_base_path)

    # generate phantom
    #os.system('python phantomgen_vectorized.py sub-%.3d_ses-1/' % sub_id)

    # remove noise
    #os.system('dwidenoise %s/dwi-SNR=12.nii %s/dwi-denoised.nii -noise %s/sigmas.nii -force' % (subject_base_path,subject_base_path,subject_base_path))

    # apply gudbjartsson correction
    #os.system('mrcalc %s/dwi-denoised.nii 2 -pow %s/sigmas.nii 2 -pow -sub 0.001 -max -sqrt %s/dwi-corrected.nii -force' % (subject_base_path,subject_base_path,subject_base_path))

    # fit DTI
    os.system('mkdir %s/dti' % subject_base_path)
    os.system('dti %s/dwi-corrected.nii %s/Penthera_3T.txt %s/dti/results.nii -mask %s/mask.nii -response 0 -correction 0 -fa -md' % (subject_base_path,subject_base_path,subject_base_path,subject_base_path))

    # get lambdas from DTI
    lambda1, lambda23 = read_lambdas('%s/dti/results_DTInolin_ResponseAnisotropic.txt' % subject_base_path)

    # fit MRDS
    os.system('mkdir %s/mrds' % subject_base_path)
    os.system('mdtmrds %s/dwi-corrected.nii %s/Penthera_3T.txt %s/mrds/results.nii -correction 0 -response %.9f,%.9f -mask %s/mask.nii -modsel bic -each -intermediate -fa -md -mse -method diff 1' % (subject_base_path,subject_base_path,subject_base_path,lambda1,lambda23,subject_base_path))

    # get tensors from MRDS
    os.system('python ../mrds2tensors.py %s/mrds/results_MRDS_Diff_BIC_EIGENVALUES.nii %s/mrds/results_MRDS_Diff_BIC_COMP_SIZE.nii %s/mrds/results_MRDS_Diff_BIC_PDDs_CARTESIAN.nii %s/mrds/results_MRDS_Diff_BIC_TENSOR.nii' % (subject_base_path,subject_base_path,subject_base_path,subject_base_path))


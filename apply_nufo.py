import numpy as np
import nibabel as nib
import sys, itertools

MRDS_RESULTS_PATH = sys.argv[1]
MRDS_PREFIX = sys.argv[2]
MRDS_METHOD = sys.argv[3]
MRDS_N = int(sys.argv[4])
MOSEMAP_FILENAME  = sys.argv[5]

print('[INFO] Loading input files...')
compsize    = [nib.load( '%s/%s_MRDS_%s_V%d_COMP_SIZE.nii'      % (MRDS_RESULTS_PATH,MRDS_PREFIX,MRDS_METHOD,N) ).get_fdata() for N in range(1,MRDS_N+1)]
eigenvalues = [nib.load( '%s/%s_MRDS_%s_V%d_EIGENVALUES.nii'    % (MRDS_RESULTS_PATH,MRDS_PREFIX,MRDS_METHOD,N) ).get_fdata() for N in range(1,MRDS_N+1)]
numcomp     = [nib.load( '%s/%s_MRDS_%s_V%d_NUM_COMP.nii'       % (MRDS_RESULTS_PATH,MRDS_PREFIX,MRDS_METHOD,N) ).get_fdata() for N in range(1,MRDS_N+1)]
pdds        = [nib.load( '%s/%s_MRDS_%s_V%d_PDDs_CARTESIAN.nii' % (MRDS_RESULTS_PATH,MRDS_PREFIX,MRDS_METHOD,N) ).get_fdata() for N in range(1,MRDS_N+1)]
isotropic   = [nib.load( '%s/%s_MRDS_%s_V%d_ISOTROPIC.nii'      % (MRDS_RESULTS_PATH,MRDS_PREFIX,MRDS_METHOD,N) ).get_fdata() for N in range(1,MRDS_N+1)]
mse         = [nib.load( '%s/%s_MRDS_%s_V%d_MSE.nii'            % (MRDS_RESULTS_PATH,MRDS_PREFIX,MRDS_METHOD,N) ).get_fdata() for N in range(1,MRDS_N+1)]
fa          = [nib.load( '%s/%s_MRDS_%s_V%d_FA.nii'             % (MRDS_RESULTS_PATH,MRDS_PREFIX,MRDS_METHOD,N) ).get_fdata() for N in range(1,MRDS_N+1)]
md          = [nib.load( '%s/%s_MRDS_%s_V%d_MD.nii'             % (MRDS_RESULTS_PATH,MRDS_PREFIX,MRDS_METHOD,N) ).get_fdata() for N in range(1,MRDS_N+1)]

aux = nib.load( '%s/%s_MRDS_%s_V1_EIGENVALUES.nii' % (MRDS_RESULTS_PATH,MRDS_PREFIX,MRDS_METHOD) )
X,Y,Z  = aux.shape[0:3]
affine = aux.affine
header = aux.header
voxels = itertools.product( range(X), range(Y), range(Z) )

print('[INFO] Loading nufo image...')
mosemap = nib.load(MOSEMAP_FILENAME).get_fdata().astype(np.uint8)

compsize_output    = np.zeros((X,Y,Z,3), dtype=np.float32)
eigenvalues_output = np.zeros((X,Y,Z,9), dtype=np.float32)
numcomp_output     = np.zeros((X,Y,Z),   dtype=np.float32)
pdds_output        = np.zeros((X,Y,Z,9), dtype=np.float32)
isotropic_output   = np.zeros((X,Y,Z,2), dtype=np.float32)
mse_output         = np.zeros((X,Y,Z),   dtype=np.float32)
fa_output          = np.zeros((X,Y,Z,3), dtype=np.float32)
md_output          = np.zeros((X,Y,Z,3), dtype=np.float32)

print('[INFO] Selecting images with mosemap')
for (x,y,z) in voxels:
    N = mosemap[x,y,z]-1

    if N > -1:
        compsize_output[x,y,z,:]    = compsize[N][x,y,z,:]
        eigenvalues_output[x,y,z,:] = eigenvalues[N][x,y,z,:]
        numcomp_output[x,y,z]       = numcomp[N][x,y,z]
        pdds_output[x,y,z,:]        = pdds[N][x,y,z,:]
        isotropic_output[x,y,z,:]   = isotropic[N][x,y,z,:]
        mse_output[x,y,z]           = mse[N][x,y,z]
        fa_output[x,y,z,:]          = fa[N][x,y,z,:]
        md_output[x,y,z,:]          = md[N][x,y,z,:]

print('[INFO] Writing output files...')
nib.save( nib.Nifti1Image(compsize_output,    affine, header), '%s/%s_MRDS_%s_NUFO_COMP_SIZE.nii'      % (MRDS_RESULTS_PATH,MRDS_PREFIX,MRDS_METHOD) )
nib.save( nib.Nifti1Image(eigenvalues_output, affine, header), '%s/%s_MRDS_%s_NUFO_EIGENVALUES.nii'    % (MRDS_RESULTS_PATH,MRDS_PREFIX,MRDS_METHOD) )
nib.save( nib.Nifti1Image(numcomp_output,     affine, header), '%s/%s_MRDS_%s_NUFO_NUM_COMP.nii'       % (MRDS_RESULTS_PATH,MRDS_PREFIX,MRDS_METHOD) )
nib.save( nib.Nifti1Image(pdds_output,        affine, header), '%s/%s_MRDS_%s_NUFO_PDDs_CARTESIAN.nii' % (MRDS_RESULTS_PATH,MRDS_PREFIX,MRDS_METHOD) )
nib.save( nib.Nifti1Image(isotropic_output,   affine, header), '%s/%s_MRDS_%s_NUFO_ISOTROPIC.nii'      % (MRDS_RESULTS_PATH,MRDS_PREFIX,MRDS_METHOD) )
nib.save( nib.Nifti1Image(mse_output,         affine, header), '%s/%s_MRDS_%s_NUFO_MSE.nii'            % (MRDS_RESULTS_PATH,MRDS_PREFIX,MRDS_METHOD) )
nib.save( nib.Nifti1Image(fa_output,          affine, header), '%s/%s_MRDS_%s_NUFO_FA.nii'             % (MRDS_RESULTS_PATH,MRDS_PREFIX,MRDS_METHOD) )
nib.save( nib.Nifti1Image(md_output,          affine, header), '%s/%s_MRDS_%s_NUFO_MD.nii'             % (MRDS_RESULTS_PATH,MRDS_PREFIX,MRDS_METHOD) )

print('Done!')
    
import numpy as np
import sys, os
"""
This script take the myelo_inferno pre-processed data (the output of tractoflow) and create a folder
with a subject for every session of every subject.
"""

sub_ids = np.arange(start=3, stop=27, step=1)
ses_ids = np.arange(start=1, stop=2, step=1)

data_basepath       = '/home/local/USHERBROOKE/here2602/SCIL/braindata/databases/CHUS/myelo_inferno/derivatives/reodered_data/output/'
tractoflow_basepath = '/home/local/USHERBROOKE/here2602/SCIL/braindata/databases/CHUS/myelo_inferno/derivatives/tractoflow/output/'

output_basepath = sys.argv[1]

for sub_id in sub_ids:
    for ses_id in ses_ids:
        subject_name = 'sub-%.3d_ses-%d' % (sub_id,ses_id)
        subject_path = '%s/%s' % (output_basepath, subject_name)

        print('[INFO] Getting data for %s/' % (subject_name))

        """# create subject folder
        os.system('mkdir %s' % (subject_path))

        # copy bvec and bval files
        os.system('cp %s/sub-%.3d/ses-%d/dwi/%s_dwi.bvec %s/bvec' % (data_basepath, sub_id, ses_id, subject_name, subject_path))
        os.system('cp %s/sub-%.3d/ses-%d/dwi/%s_dwi.bval %s/bval' % (data_basepath, sub_id, ses_id, subject_name, subject_path))

        # copy dwi image
        os.system('cp %s/%s/Resample_DWI/%s__dwi_resampled.nii.gz %s/dwi.nii.gz' % (tractoflow_basepath, subject_name, subject_name, subject_path))
        os.system('cp %s/%s/Resample_B0/%s__b0_mask_resampled.nii.gz %s/mask.nii.gz' % (tractoflow_basepath, subject_name, subject_name, subject_path))

        # uncompress images for mrds
        os.system('gzip -d %s/dwi.nii.gz'  % (subject_path))
        os.system('gzip -d %s/mask.nii.gz' % (subject_path))"""
        
        # merge bvec and bval files into scheme file
        os.system('python3 /home/local/USHERBROOKE/here2602/SCIL/scripts/bvalbvec2scheme.py %s/bval %s/bvec %s/scheme 1' % (subject_path, subject_path, subject_path))

print('[INFO] Done!')
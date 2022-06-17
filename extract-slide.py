import numpy as np
import nibabel as nib
import sys

dwi_filename = sys.argv[1]
axis = sys.argv[2]
slide = int(sys.argv[3])
output_filename = sys.argv[4]

print('[INFO] Loading DWI image')
dwi = nib.load( dwi_filename )
data = dwi.get_fdata()
X,Y,Z,N = data.shape

print('[INFO] Creating ROI')
roi = np.zeros( (X,Y,Z), dtype=np.uint8 )

if axis == 'axial':
    roi[:,:,slide] = np.ones((X,Y), dtype=np.uint8)
elif axis == 'sagittal':
    roi[slide,:,:] = np.ones((Y,Z), dtype=np.uint8)
elif axis == 'coronal':
    roi[:,slide,:] = np.ones((X,Z), dtype=np.uint8)
else:
    print('[ERROR] Axis choice not supported!')

nib.save( nib.Nifti1Image(roi, dwi.affine, dwi.header), output_filename )

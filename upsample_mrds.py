import numpy as np
import nibabel as nib
import sys, itertools, queue

def valid(i,j,k, X,Y,Z):


eigenvalues_filename = sys.argv[1]
compsize_filename = sys.argv[2]
pdds_filename = sys.argv[3]

mrds_results_path = sys.argv[1]

eigenvalues = []
compsize = []
pdds = []

combinations = [
    [],
    [(0,1), (1,0)],
    [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,1,0), (2,0,1)]
]

di = [-1, 0, 1, 0]
dj = [0, 1, 0, -1]

X,Y,Z = compsize.shape
voxels = itertools.product( range(X), range(Y), range(Z) )

for N in [2, 3]:
    eigenvalues_filename = 'results_MRDS_Diff_V%d_EIGENVALUES.nii' % N
    compsize_filename = 'results_MRDS_Diff_V%d_COMP_SIZE.nii' % N
    pdds_filename = 'results_MRDS_Diff_V%d_PDDs_CARTESIAN.nii' % N

    eigenvalues_file = nib.load( eigenvalues_filename )
    compsize_file = nib.load( compsize_filename )
    pdds_file = nib.load( pdds_filename )

    eigenvalues.append( eigenvalues_file.get_fdata() )
    compsize.append( compsize_file.get_fdata() )
    pdds.append( pdds_file.get_fdata() )

    # start pairing tensors
    idx = np.zeros( (X,Y,Z, N), dtype=np.uint8 ) # ids of the paired tensors
    vis = np.zeros( (X,Y,Z),    dtype=np.uint8 ) # visited tensors

    Q = queue.Queue()
    queue.put( voxels[0] )

    while Q.empty() == False:
        (x,y,z) = Q.get()

        # check for every neighbor if we can extend the voxel
        for k in len(di):

            min_angle = 1e9

            for i in range(0,N):
                (a,b) = combinations[N][i]

                dir1 = pdds[x,y,z, ]

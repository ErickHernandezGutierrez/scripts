import numpy as np
import sys
from utils import load_scheme


scheme_filename = sys.argv[1]

bvecs,bvals = load_scheme(scheme_filename)

bvals_count = {}

for i,bval in enumerate(bvals):
    if np.abs(bval) < 1e-3:
        print(i)
    if bval in bvals_count:
        bvals_count[bval] = bvals_count[bval]+1
    else:
        bvals_count[bval] = 1

for bval in bvals_count:
    print('bval = %f : %d' % (bval, bvals_count[bval]))

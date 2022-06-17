import sys, os

scheme_filename = sys.argv[1]
bvals_filename  = sys.argv[2]
bvecs_filename  = sys.argv[3]
if len(sys.argv) > 4:
    invert_x = int(sys.argv[4])
else:
    invert_x = False

print('Generating bval bvec from %s' % scheme_filename)

try:
    bvals, bvecs = [], []

    # read content of the input file
    print('[INFO] Reading scheme file...')
    scheme_file = open(scheme_filename, 'rt')
    lines = scheme_file.readlines()
    for line in lines:
        x,y,z,b = line.split(' ')
        x,y,z,b = float(x), float(y), float(z), float(b)

        bvals.append( b )
        bvecs.append( [x,y,z] )
    scheme_file.close()

    # write bvals file
    print('[INFO] Writing bval file...')
    bvals_file = open(bvals_filename, 'wt')
    for b in bvals:
        bvals_file.write('%lf ' % b)
    bvals_file.write('\n')
    bvals_file.close()

    # write bvecs file
    print('[INFO] Writing bvec file...')
    bvecs_file = open(bvecs_filename, 'wt')
    for i in range(3):
        for g in bvecs:
            if i==0 and invert_x==1:
                bvecs_file.write('%lf ' % -g[i])
            else:
                bvecs_file.write('%lf ' % g[i])
        bvecs_file.write('\n')
    bvecs_file.close()

    print('[INFO] Done!')
except:
    print('[ERROR] Could not open scheme file or create bval/bvec files!')


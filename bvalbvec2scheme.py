import sys, os

bvals_filename  = sys.argv[1]
bvecs_filename  = sys.argv[2]
scheme_filename = sys.argv[3]
if len(sys.argv) > 4:
    invert_x = bool(sys.argv[4])
else:
    invert_x = False

print('Generating scheme file from %s and %s' % (bvals_filename, bvecs_filename))

try:
    # open files
    bvals_file = open(bvals_filename, 'rt')
    bvecs_file = open(bvecs_filename, 'rt')

    bvals = []
    bvecs = []

    print('[INFO] Reading bval and bvec files...')

    # read bvals
    line = bvals_file.readlines()[0]
    bvals = [float(value) for value in line.split(' ')]

    # read bvecs
    lines = bvecs_file.readlines()
    bvecs.append( [float(value) for value in lines[0].split(' ')] )
    bvecs.append( [float(value) for value in lines[1].split(' ')] )
    bvecs.append( [float(value) for value in lines[2].split(' ')] )

    # close inputs files
    bvals_file.close()
    bvecs_file.close()

    # create scheme file
    scheme_file = open(scheme_filename, 'wt')

    print('[INFO] Writing scheme file...')
    for i in range(len(bvals)):
        if invert_x:
            scheme_file.write('%lf %lf %lf %lf\n' % (-bvecs[0][i], bvecs[1][i], bvecs[2][i], bvals[i]))
        else:
            scheme_file.write('%lf %lf %lf %lf\n' % (bvecs[0][i], bvecs[1][i], bvecs[2][i], bvals[i]))

    scheme_file.close()

    print('[INFO] Done!')
except:
    print('[ERROR] Could not open bval or bvec files or create scheme file!')


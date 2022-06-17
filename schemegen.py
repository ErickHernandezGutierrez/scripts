import sys, os

subject_folder = sys.argv[1]

print('[INFO] Generating scheme for %s...' % subject_folder)

bvals_filename  = subject_folder + '/bval'
bvecs_filename  = subject_folder + '/bvec'
scheme_filename = subject_folder + '/mrds.scheme'

try:
    # open files
    bvals_file = open(bvals_filename, 'rt')
    bvecs_file = open(bvecs_filename, 'rt')

    bvals = []
    bvecs = []

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

    # create scheme
    scheme_file = open(scheme_filename, 'wt')

    print('[WARNING] Changing x to -x.')
    for i in range(len(bvals)):
        scheme_file.write('%.19lf %.19lf %.19lf %.3lf\n' % (-bvecs[0][i], bvecs[1][i], bvecs[2][i], bvals[i]))
        #scheme_file.write('%.19lf %.19lf %.19lf %.3lf\n' % (bvecs[0][i], bvecs[1][i], bvecs[2][i], bvals[i]))

    scheme_file.close()

    print('[INFO] Done!')
except:
    print('[ERROR] Could not open bval or bvec files! or subject directory does not exist')


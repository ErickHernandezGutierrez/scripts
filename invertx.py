import numpy as np
import sys

bvecs_filename = sys.argv[1]
output_filename = sys.argv[2]

bvecs = []

# open files
bvecs_file = open(bvecs_filename, 'rt')

print('[INFO] Reading bvec file...')

# read bvecs
lines = bvecs_file.readlines()
bvecs.append( [float(value) for value in lines[0].split(' ') if value!='\n'] )
bvecs.append( [float(value) for value in lines[1].split(' ') if value!='\n'] )
bvecs.append( [float(value) for value in lines[2].split(' ') if value!='\n'] )

# close inputs files
bvecs_file.close()

# create scheme file
output_file = open(output_filename, 'wt')

print('[INFO] Inverting axis X from file %s' % (bvecs_filename))
for i in range(3):
    for val in bvecs[i]:
        if i == 0:
            output_file.write( '%lf ' % (-val) )
        else:
            output_file.write( '%lf ' % (val) )
    output_file.write( '\n' )

output_file.close()

print('[INFO] Done!')


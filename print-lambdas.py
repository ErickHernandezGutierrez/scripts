import numpy as np
import sys

def read_lambdas(lambdas_filename):
    with open(lambdas_filename, 'rt') as lambdas_file:
        line = lambdas_file.readlines()[0]
        lambdas = line.split(' ') [0:3]
        lambdas = [float(value) for value in lambdas]

               #lambda1    #lambda23
        return lambdas[0], (lambdas[1]+lambdas[2])/2.0

ini = int(sys.argv[1])
end = int(sys.argv[2])

IDs = np.arange(ini, end)

lambdas1  = []
lambdas23 = []

for subject_id in IDs:
    subject = './sub-%.3d_ses-1' % subject_id

    lambdas = read_lambdas('%s/dti/results_DTInolin_ResponseAnisotropic.txt' % subject )

    lambdas1.append( lambdas[0] )
    lambdas23.append( lambdas[1] )

for lambda1 in lambdas1:
    print(lambda1, end=' ')

print()

for lambda23 in lambdas23:
    print(lambda23, end=' ')

print()
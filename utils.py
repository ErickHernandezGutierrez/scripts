import numpy as np

def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r  = np.hypot(hxy, z)
    theta = np.arctan2(z, hxy)
    if x>0:
        phi = np.arctan2(y, x)
    elif x<0 and y>=0:
        phi = np.arctan2(y, x)+np.pi
    elif x<0 and y<0:
        phi = np.arctan2(y, x)-np.pi
    elif np.abs(x<1e-6) and y>0:
        phi = np.pi/2
    elif np.abs(x<1e-6) and y<0:
        phi = -np.pi/2
    else:
        phi=np.inf
    
    return np.array( [theta, phi, r] )

def load_scheme(scheme_filename, bval=None):
    bvecs, bvals, idx = [], [], []

    scheme_file = open(scheme_filename, 'rt')

    for i,line in enumerate(scheme_file.readlines()):
        x,y,z,b = line.split(' ')
        x,y,z,b = float(x),float(y),float(z),float(b)
        bvecs.append( np.array([x,y,z]) )
        bvals.append( b )
        if bval!=None and np.abs(bval-b) < 1e-1:
            idx.append(i)

    if bval==None:
        return np.array(bvecs), np.array(bvals)
    else:
        return np.array(bvecs), np.array(bvals), idx

def load_lambdas(lambdas_filename):
    with open(lambdas_filename, 'rt') as lambdas_file:
        line = lambdas_file.readlines()[0]
        lambdas = line.split(' ') [0:3]
        lambdas = [float(value) for value in lambdas]

               #lambda1    #lambda2    #lambda3 
        return lambdas[0], lambdas[1], lambdas[2]

def save_scheme(bvecs, bvals, scheme_filename):
    with open(scheme_filename, 'wt') as scheme_file:
        for (bvec, bval) in zip(bvecs, bvals):
            scheme_file.write('%f %f %f %f\n' % (bvec[0], bvec[1], bvec[2], bval))

def load_dirs(dirs_filename, ndirs=500):
    directions = np.fromfile(dirs_filename, dtype=np.float64)
    directions = np.reshape(directions, (ndirs, 3))

    return directions

def lambdas2fa(lambdas):
    a = np.sqrt(0.5)
    b = np.sqrt( (lambdas[0]-lambdas[1])**2 + (lambdas[1]-lambdas[2])**2 + (lambdas[2]-lambdas[0])**2 )
    c = np.sqrt( lambdas[0]**2 + lambdas[1]**2 + lambdas[2]**2 )

    return a*b/c

def getSkewSymmetricMatrixFromVector(v):
    return np.array([
        [ 0,    -v[2],  v[1]],
        [ v[2],  0,    -v[0]],
        [-v[1],  v[0],  0]
    ])

# a = axis reference
def getRotationFromDir(a, dir):
    #a = (0,1,0)
    v = np.cross(a, dir)
    c = np.dot(a, dir)

    V = getSkewSymmetricMatrixFromVector(v)

    return np.identity(3) + V + V@V * (1/(1+c))

def tensor2matrix(tensor):
    D = np.zeros( (3,3) )

    D[0,0] = tensor[0]
    D[1,1] = tensor[1]
    D[2,2] = tensor[2]
    D[0,1],D[1,0] = tensor[3], tensor[3]
    D[0,2],D[2,0] = tensor[4], tensor[4]
    D[1,2],D[2,1] = tensor[5], tensor[5]

    return D

def invariants(tensor):
    D = tensor2matrix(tensor)

    I1 = D[0][0] + D[1][1] + D[2][2]
    I2 = D[0][0]*D[1][1] + D[1][1]*D[2][2] + D[2][2]*D[0][0] - (D[0][1]*D[0][1] + D[0][2]*D[0][2] + D[1][2]*D[1][2])
    I3 = D[0][0]*D[1][1]*D[2][2] + 2*D[0][1]*D[0][2]*D[1][2] - (D[2][2]*D[0][1]*D[0][1] + D[1][1]*D[0][2]*D[0][2] + D[0][0]*D[1][2]*D[1][2])
    I4 = D[0][0]*D[0][0] + D[1][1]*D[1][1] + D[2][2]*D[2][2] + 2*(D[0][1]*D[0][1] + D[0][2]*D[0][2] + D[1][2]*D[1][2])

    return np.array([I1, I2, I3, I4])

def eigenvalues(tensor):
    I = invariants(tensor)

    v = (I[0]/3.0)**2 - I[1]/3.0
    s = (I[0])**3 - I[0]*I[1]/6.0 + I[2]/2.0
    o = np.arccos( clamp(s/pow(v, 3.0/2.0), min_value=-1, max_value=1) ) / 3

    lambda1 = I[0]/3 + 2*np.sqrt(v)*np.cos(o)
    lambda2 = I[0]/3 - 2*np.sqrt(v)*np.cos(np.pi/3 + o)
    lambda3 = I[0]/3 - 2*np.sqrt(v)*np.cos(np.pi/3 - o)

    return np.array([lambda1, lambda2, lambda3])

def eigenvectors(tensor):
    D = tensor2matrix(tensor)

    lambdas = eigenvalues(tensor)

    A = np.full(3, D[0,0]) - lambdas
    B = np.full(3, D[1,1]) - lambdas
    C = np.full(3, D[2,2]) - lambdas

    e = np.zeros((3,3))
    for i in range(3):
        e[i,:] = np.array([
            (D[0,1]*D[1,2] - B[i]*D[0,2])*(D[0,2]*D[1,2] - C[i]*D[0,1]),
            (D[0,2]*D[1,2] - C[i]*D[0,1])*(D[0,1]*D[0,2] - A[i]*D[1,2]),
            (D[0,1]*D[0,2] - A[i]*D[1,2])*(D[0,1]*D[1,2] - B[i]*D[0,2])
        ])

        e[i,:] = e[i,:] / np.linalg.norm(e[i,:])

    return e

def tensor2fa(tensor):
    I = invariants(tensor)

    return np.sqrt(1 - I[1]/I[3])

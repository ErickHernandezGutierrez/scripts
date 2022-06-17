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


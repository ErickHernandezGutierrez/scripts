from random import random
import numpy as np
import itertools

SNRs = [30, 12, np.inf]

SNR = np.inf

def lambdas2fa(lambdas):
    a = np.sqrt(0.5)
    b = np.sqrt( (lambdas[0]-lambdas[1])**2 + (lambdas[1]-lambdas[2])**2 + (lambdas[2]-lambdas[0])**2 )
    c = np.sqrt( lambdas[0]**2 + lambdas[1]**2 + lambdas[2]**2 )

    return a*b/c

def random_angles(a, b, size):
    return a + (b-a)*np.random.rand(size)

# Convert from spherical ccordinates to cartesian coordinates
def sph2cart(r, azimuth, zenith):
    x = r * np.cos(azimuth) * np.sin(zenith)
    y = r * np.sin(azimuth) * np.sin(zenith)
    z = r * np.cos(zenith)

    return np.array([x,y,z]) #"""

# Compute a rotation matrix corresponding to the orientation (azimuth,zenith)
#     azimuth (phi):	angle in the x-y plane
#     zenith  (theta):	angle in the x-z plane
# ---------------------------------------------------------------------------
def get_rotation(azimuth, zenith):
    azimuth = azimuth % (2*np.pi)
    zenith  = zenith % (np.pi)

    azimuth_rotation = np.array([
        np.array([np.cos(azimuth), -np.sin(azimuth), 0.0]),
        np.array([np.sin(azimuth),  np.cos(azimuth), 0.0]),
        np.array([0.0,              0.0,             1.0])
    ])

    zenith_rotation = np.array([
        np.array([ np.cos(zenith), 0.0, np.sin(zenith)]),
        np.array([ 0.0,            1.0,            0.0]),
        np.array([-np.sin(zenith), 0.0, np.cos(zenith)])
    ]) #"""

    """zenith_rotation = np.array([
        np.array([ 1.0,            0.0,             0.0]),
        np.array([ 0.0, np.cos(zenith), -np.sin(zenith)]),
        np.array([ 0.0, np.sin(zenith),  np.cos(zenith)])
    ]) #"""

    #return np.matmul(azimuth_rotation, zenith_rotation) # primero aplica la azimuth
    return np.matmul(zenith_rotation, azimuth_rotation) # primero aplica la zenith

    #return azimuth_rotation @ zenith_rotation
    #return zenith_rotation @ azimuth_rotation

def get_scale(u, v, kappa):
    from scipy.special import hyp1f1 as M

    return (1/M(1.0/2.0,3.0/2.0,kappa)) * np.exp( kappa*(u@v)**2 )

# Load the sampling scheme
def load_scheme(scheme_filename, bval=None):
    scheme = []

    scheme_file = open(scheme_filename, 'rt')

    for line in scheme_file.readlines():
        x,y,z,b = line.split(' ')
        scheme.append( [float(x),float(y),float(z),float(b)] )

    return np.array( scheme )

# Probe the signal at a given q-space coordinate for a given voxel configuration
# ------------------------------------------------------------------------------
def E(g, b, voxel, dispersion=False, noise=False):
    ntensors = voxel['n']
    pdds = voxel['pdds']
    lambdas = voxel['eigenvals']
    ndirs = voxel['ndirs']

    signal = 0
    for i in range(ntensors):
        azimuth, zenith = pdds[2*i], pdds[2*i+1]
        R = get_rotation(azimuth, zenith)

        if dispersion:
            dirs,weights = get_dispersion(azimuth)

            aux = 0
            for j in range(ndirs):
                R = rotations[i,j]  #direction
                w = weights[i,j]    #weight of the direction
                D = R @ np.diag(lambdas[i]) @ R.transpose()
                aux = aux + w*np.exp( -b*(g.transpose()@D@g) )
            signal = signal + alphas[i] * aux
        else:
            R = rotations[i]
            D = R @ np.diag(lambdas[i]) @ R.transpose()
            signal = signal + alphas[i] * np.exp( -b*(g.transpose()@D@g) )

    return signal

def get_acquisition(voxel, scheme, sigma=0.0):
    nsamples = len(scheme)
    signal = np.zeros(nsamples)

    for i in range(nsamples):
        signal[i] = E(g=scheme[i][0:3], b=scheme[i][3], voxel=voxel)
        # todo: add noise to the signal

    return signal

# Return the directions and the weights of the dispersion
# (azimuth, zenith): spherical coordinates of the PDD
# ndirs: number of directions in the dispersion
# kappa: concentration parameter
# ------------------------------------------------------------------------------
def get_dispersion(azimuth, zenith, ndirs, kappa):
    (x,y,z) = sph2cart(1.0, azimuth, zenith)

    eps_azimuth = random_angles(a=-np.pi/2, b=np.pi/2, size=ndirs)
    eps_zenith  = random_angles(a=-np.pi/2, b=np.pi/2, size=ndirs)

    points,dirs, weights = [],[],[]
    for i in range(ndirs):
        R = get_rotation(azimuth+eps_azimuth[i], zenith+eps_zenith[i])
        dirs.append( R )
        a,b = R@(x,y,z), R@(-x,-y,-z)

        s = get_scale((x,y,z), a, kappa)
        weights.append( s )
        a,b = s*a,s*b

        points = points + [a,b]

    return np.array(points), np.array(dirs), np.array(weights)

def plot_dispersion_old(points):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    X,Y,Z = points[:,0], points[:,1], points[:,2]
    ax.scatter(X, Y, Z, color='blue')

    # fake bounding box to fix axis size
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    plt.show()

def plot_dispersion(azimuth, zenith, dirs, weights):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    (x,y,z) = sph2cart(1.0, azimuth, zenith)
    points = []
    for (R,s) in zip(dirs, weights):
        a,b = R@(x,y,z), R@(-x,-y,-z)
        a,b = s*a, s*b
        points = points + [a,b]
    points = np.array(points)

    X,Y,Z = points[:,0], points[:,1], points[:,2]
    ax.scatter(X, Y, Z, color='blue')

    max_amp = 10
    (x,y,z) = (max_amp*x, max_amp*y, max_amp*z)
    ax.scatter(x, y, z, color='red')
    ax.scatter(-x, -y, -z, color='red')

    # fake bounding box to fix axis size
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    plt.show()

def plot_angles(azimuth, zenith, nangles, kappa):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(20,10))  # Square figure
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(r'$\kappa = %d$' % kappa)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    mu, sigma = 0, 0.05
    max_angle = 2
    #eps_azimuth = np.random.normal(mu, sigma, nangles)
    #eps_zenith  = np.random.normal(mu, sigma, nangles)
    #eps_azimuth = np.random.rand(nangles)*0.01
    #eps_zenith  = np.random.rand(nangles)
    eps_azimuth = random_angles(a=-np.pi/max_angle, b=np.pi/max_angle, size=nangles)
    eps_zenith  = random_angles(a=-np.pi/max_angle, b=np.pi/max_angle, size=nangles)

    line = np.array([
        np.array([0.0, 1.0,  0.0]), #endpoints
        np.array([0.0, -1.0, 0.0])
    ])

    line = line @ get_rotation(azimuth, zenith)

    #ax.plot(line[:,0], line[:,1], line[:,2], marker='o', color='red')

    points = []

    for i in range(nangles):
        R = get_rotation(azimuth+eps_azimuth[i], zenith+eps_zenith[i])
        #R = get_rotation(azimuth+eps_azimuth[i], zenith)
        #R = get_rotation(azimuth, zenith+eps_zenith[i])
        #R = get_rotation(azimuth, zenith)

        rot_line = line @ R #rotate endpoints

        S = get_scale(line, rot_line, kappa)

        rot_line = rot_line @ S #scale endpoints

        points.append( rot_line[0] )
        points.append( rot_line[1] )

        #ax.scatter(rot_line[:,0], rot_line[:,1], rot_line[:,2], marker='o', color='blue')
        #ax.plot(rot_line[:,0], rot_line[:,1], rot_line[:,2], marker='o', color='blue')

    points = np.array( points )
    X,Y,Z = points[:,0], points[:,1], points[:,2]
    ax.scatter(X, Y, Z, color='blue')

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    plt.show()

def random_fractions(n):
    if n==1:
        return np.array([1,0,0])
    elif n==2:
        alpha = np.random.rand()
        return np.array([alpha, 1-alpha, 0])
    else:
        alpha = np.random.rand()
        beta  = np.random.rand()*alpha
        return np.array([alpha, beta, 1-alpha-beta])

def random_eigenvalues(n):
    lambdas = np.zeros(9)

    for i in range(n):
        lambdas[3*i] = 0.002
        lambdas[3*i + 1] = 0.0001
        lambdas[3*i + 2] = 0.0001

    return lambdas

def random_sphere_directions(n):
    angles = np.zeros(6)
    
    for i in range(n):
        angles[2*i]     = np.random.rand() * (2*np.pi) #azimuth angle
        angles[2*i + 1] = np.random.rand() * (np.pi)   #zenith angle

    return angles


scheme = load_scheme('Penthera_3T.txt')
X,Y,Z,N = 16,16,5,len(scheme) # dimensions of the phantom
voxels = itertools.product( range(X), range(Y), range(Z) )

numcomp = np.random.randint( low=1, high=4, size=(X,Y,Z) ) # number of fascicles per voxel
fracs   = np.zeros( (X,Y,Z, 3) )                           # volume fractions per voxel
eigens  = np.zeros( (X,Y,Z, 9) )                           # eigenvalues per voxel
pdds    = np.zeros( (X,Y,Z, 6) )                           # principal diffusion directions per voxel
dwi     = np.zeros( (X,Y,Z, N) )                           # volume

"""for (x,y,z) in voxels:
    n = numcomp[x,y,z]

    fracs[x,y,z,  :] = random_fractions(n)
    eigens[x,y,z, :] = random_eigenvalues(n)
    pdds[x,y,z,   :] = random_sphere_directions(n)

    voxel = {}
    voxel['x'] = x
    voxel['y'] = y
    voxel['z'] = z
    voxel['n'] = n
    voxel['alphas'] = fracs[x,y,z, :]
    voxel['eigenvals'] = eigens[x,y,z, :]
    voxel['pdds'] = pdds[x,y,z, :]
    voxel['dispersion'] = [dirs, weights, ]

    dwi[x,y,z, :] = get_acquisition(voxel, scheme)#"""

azimuth=np.pi/2
zenith=np.pi/4
points, dirs, weights = get_dispersion(azimuth=azimuth, zenith=zenith, ndirs=6000, kappa=8)
#plot_dispersion_old(points)
plot_dispersion(azimuth=azimuth, zenith=zenith, dirs=dirs, weights=weights)
#kappa = 16
#ndirs = 6000
#plot_angles(0, 0, ndirs, kappa)

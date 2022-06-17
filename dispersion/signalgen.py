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
        np.array([ 0.0,            1.0, 0.0]),
        np.array([-np.sin(zenith), 0.0, np.cos(zenith)])
    ])

    return azimuth_rotation @ zenith_rotation

def get_scale(line, rot_line):
    print('lines')
    print(line)
    print(rot_line)
    u = (line[1] - line[0])
    u = u / np.linalg.norm(u)
    v = rot_line[1] - rot_line[0]
    v = v / np.linalg.norm(v)
    print('dirs')
    print(u)
    print(v)
    print('dot = %f' % (u@v))
    s = (u@v)**2
    #s = np.exp( (u@v)**2 )
    print(s, end='\n\n')
    return np.diag(np.array([s,s,s]))

# Load the sampling scheme
def load_scheme(scheme_filename, bval=None):
    scheme = []

    scheme_file = open(scheme_filename, 'rt')

    for line in scheme_file.readlines():
        x,y,z,b = line.split(' ')
        scheme.append( [float(x),float(y),float(z),float(b)] )

    return np.array( scheme )

# Probe the signal at a given q-space coordinate for a given voxel configuration
# ----------------------------------------------
def E(q, b, voxel):
    signal = 0

    ntensors = voxel['n']
    lambdas  = voxel['eigenvalues']
    R = voxel['rotations']
    alphas = voxel['fractions']

    for i in range(ntensors):
        D = R[i] @ np.diag(lambdas[i]) @ R[i].transpose()
        signal = signal + alphas[i] * np.exp( -b*(q.transpose()@D@q) )

    return signal

def get_acquisition(scheme, sigma, voxel):
    ndirs = len(scheme)
    signal = np.zeros(ndirs)

    for i in range(ndirs):
        signal[i] = E(scheme[i][0:3], scheme[i][3], voxel)
        # todo: add noise to the signal

    return signal

def random_angles(a, b, size):
    return a + (b-a)*np.random.rand(size)

def plot_angles(azimuth, zenith, nangles):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
    ax = fig.add_subplot(111, projection='3d')
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

    ax.plot(line[:,0], line[:,1], line[:,2], marker='o', color='red')

    for i in range(nangles):
        R = get_rotation(azimuth+eps_azimuth[i], zenith+eps_zenith[i])
        #R = get_rotation(azimuth+eps_azimuth[i], zenith)
        #R = get_rotation(azimuth, zenith+eps_zenith[i])
        #R = get_rotation(azimuth, zenith)

        rot_line = line @ R

        S = get_scale(line, rot_line)

        rot_line = rot_line @ S

        ax.plot(rot_line[:,0], rot_line[:,1], rot_line[:,2], marker='o', color='blue')

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

def random_rotations(n):
    angles = np.zeros(6)
    
    for i in range(n):
        angles[2*i]     = np.random.rand() * (2*np.pi) #azimuth angle
        angles[2*i + 1] = np.random.rand() * (np.pi)   #zenith angle

    return angles

# dimensions of the phantom
X,Y,Z = 16,16,5
voxels = itertools.product( range(X), range(Y), range(Z) )
numcomp = np.random.randint( low=1, high=4, size=(X,Y,Z) )
fractions   = np.zeros( (X,Y,Z, 3) )
eigenvalues = np.zeros( (X,Y,Z, 9) )
rotations   = np.zeros( (X,Y,Z, 6) )

for (x,y,z) in voxels:
    n = numcomp[x,y,z]

    fractions[x,y,z, :]   = random_fractions(n)
    eigenvalues[x,y,z, :] = random_eigenvalues(n)
    rotations[x,y,z, :]   = random_rotations(n)

theta = np.pi/4
voxel = {}
voxel['n'] = 2
voxel['eigenvalues'] = [
    np.array([0.002, 0.0001, 0.0001]),
    np.array([0.003, 0.0001, 0.0001])
]
voxel['fractions'] = [0.5, 0.5]
voxel['rotations'] = np.array([
    get_rotation(np.pi, 0),
    get_rotation(np.pi, theta)
])

#print(lambdas2fa(voxel['eigenvalues'][0]))
#print(lambdas2fa(voxel['eigenvalues'][1]))

scheme = load_scheme('Penthera_3T.txt')

#print( E(scheme[0][0:3], scheme[0][3], voxel) )
#print( get_acquisition(scheme, 0.0, voxel) )

plot_angles(0, 0, 50)

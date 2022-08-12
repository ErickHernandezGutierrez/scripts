from random import random
import numpy as np
import itertools

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R

def rotate_vec(vec, azimuth):
    rotation_vector = azimuth * np.array([0,0,1])
    rotation = R.from_rotvec(rotation_vector)

    rotated_vec = rotation.apply(vec)

    return rotated_vec

def sph2cart(r, azimuth, zenith):
    """x = r * np.cos(zenith) * np.sin(azimuth)
    y = r * np.sin(zenith) * np.sin(azimuth)
    z = r * np.cos(azimuth)
    return np.array([x,y,z]) #"""

    x = r * np.cos(azimuth) * np.sin(zenith)
    y = r * np.sin(azimuth) * np.sin(zenith)
    z = r * np.cos(zenith)
    return np.array([x,y,z]) #"""

def get_rotation(azimuth, zenith):
    azimuth = azimuth % (2*np.pi)
    zenith  = zenith % (np.pi)

    azimuth_rotation = np.matrix([
        [np.cos(azimuth), -np.sin(azimuth), 0.0],
        [np.sin(azimuth),  np.cos(azimuth), 0.0],
        [0.0,              0.0,             1.0]
    ])

    zenith_rotation = np.matrix([
        [ np.cos(zenith), 0.0, np.sin(zenith)],
        [ 0.0,            1.0,            0.0],
        [-np.sin(zenith), 0.0, np.cos(zenith)]
    ]) #"""

    """zenith_rotation = np.matrix([
        [ 1.0,            0.0,             0.0],
        [ 0.0, np.cos(zenith), -np.sin(zenith)],
        [ 0.0, np.sin(zenith),  np.cos(zenith)]
    ]) #"""

    #return azimuth_rotation
    #return zenith_rotation

    #return np.matmul(azimuth_rotation, zenith_rotation) # primero aplica la azimuth
    return np.matmul(zenith_rotation, azimuth_rotation) # primero aplica la zenith

def get_trace(radians, npoints):
    points = []

    angles = np.linspace(0, radians, npoints)

    for angle in angles:
        points.append( sph2cart(1, 0, angle) )

    return np.array(points)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

p = np.array([0, 0,  1])
q = np.array([0, 0, -1])
a = sph2cart(1, 0, np.pi/2)

npoints = 10
points = get_trace(np.pi/2, npoints)

M = get_rotation(0, np.pi/2)

P = M @ p
Q = M @ q
#p = np.matmul(M, p)
#q = np.matmul(M, q)
#p = rotate_vec(p, np.pi/2)
#q = rotate_vec(q, np.pi/2)

ax.scatter(P[0,0], P[0,1], P[0,2], color='blue', marker='+')
ax.scatter(Q[0,0], Q[0,1], Q[0,2], color='red', marker='+')
ax.scatter(p[0], p[1], p[2], color='blue', alpha=0.5)
ax.scatter(q[0], q[1], q[2], color='red', alpha=0.5)
#ax.scatter(a[0], a[1], a[2], color='green', marker='*')
ax.scatter(points[0,0], points[0,1], points[0,2], color='green', marker='o')
ax.scatter(points[1:npoints-1,0], points[1:npoints-1,1], points[1:npoints-1,2], color='green', marker='*')
#ax.scatter(points[npoints-1,0], points[npoints-1,1], points[npoints-1,2], color='green', marker='+')

plt.show()
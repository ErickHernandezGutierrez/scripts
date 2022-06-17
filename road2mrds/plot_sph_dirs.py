import matplotlib.pyplot as plt
import numpy as np
import sys
from utils import load_directions


DIRS_FILENAME = sys.argv[1]

dirs = load_directions( DIRS_FILENAME )

fig = plt.figure(figsize=(4,4))

ax = fig.add_subplot(111, projection='3d')
ax.plot(dirs[:,0], dirs[:,1], dirs[:,2], linewidth=0, marker='o', markersize=2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

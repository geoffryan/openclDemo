import sys
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)

for filename in sys.argv[1:]:

    N, time = np.loadtxt(filename, unpack=True, usecols=[0, 1])
    name = "_".join(filename.split("_")[:-1])
    ax.plot(N, time, marker='o', ls='-', lw=1, label=name)

ax.legend()

ax.set(xlabel=r'$N$', xscale='log',
       ylabel=r'Kernel Execution Time (s)', yscale='log')

fig.tight_layout()

plt.show()

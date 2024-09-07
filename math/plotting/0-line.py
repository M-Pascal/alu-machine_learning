#!usr/bin/env python3

# Importing libraries.
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3
x = np.arange(0, 11)

plt.plot(x, y, color='red')

# Illustrating x-axis, y-axis, and naming the title of graph.
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Graph')
plt.xlim(left=0, right=10)

plt.show()

import matplotlib.pyplot as plt
import numpy as np

# basic line plot
myarray = np.array([1, 2, 3])
plt.plot(myarray)
plt.xlabel('some x-axis')
plt.ylabel('some y-axis')
plt.show()

# basic scatter plot
x = np.array([1, 2, 3])
y = np.array([2, 4, 6])
plt.scatter(x, y)
plt.xlabel('some x-axis')
plt.ylabel('some y-axis')
plt.show()

import numpy as np
import pandas as pd

# series
myarray = np.array([1, 2, 3])
rownames = ['a', 'b', 'c']
myseries = pd.Series(myarray, index=rownames)
print(myseries[0])
print(myseries['a'])
print(myseries)

# dataframe
myarray = np.array([[1, 2, 3], [4, 5, 6]])
rownames = ['a', 'b']
colnames = ['one', 'two', 'three']
mydataframe = pd.DataFrame(myarray, index=rownames, columns=colnames)
print(mydataframe)
print("method 1:")
print("{}".format(mydataframe['one']))
print("method 2:")
print("{}".format(mydataframe.one))


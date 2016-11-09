import numpy as np

# access values
mylist = [[1, 2, 3], [3, 4, 5]]
myarray = np.array(mylist)
print(myarray)
print(myarray.shape)
print("First row: {}".format(myarray[0]))
print("Last row: {}".format(myarray[-1]))
print("Specific row and column: {}".format(myarray[0, 2]))
print("Whole column: {}".format(myarray[:, 2]))

# arithmetic
myarray1 = np.array([2, 2, 2])
myarray2 = np.array([3, 3, 3])
print("Addition: {}".format(myarray1 + myarray2))
print("Multiplication: {}".format(myarray1 * myarray2))
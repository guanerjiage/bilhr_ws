#!/usr/bin/env python
# Author: Erjiage GUAN, Zhifan NI, Lei WANG, Zhaoxiong WEI
import numpy as np


a = np.random.randint(0, 100, (50, 20, 10))
print(a.shape)
print(np.mean(a, axis=0).shape)
print(np.mean(a, axis=1).shape)
print(np.mean(a, axis=2).shape)

a = [[[1, 2, 3, 4], [2, 3, 4, 5]], [[3, 4, 5, 6], [5, 6, 7, 8]]]
b = np.array(a)
print(b)
print(b.flatten())
print(b.reshape((2, -1)).T)

c = np.array([11, 12, 21, 22])
print(c.reshape((2, 2)))

d = np.array([[1, 1], [2, 2], [3, 3]])
e = np.array([[3, 4, 5], [4, 5, 6]])
print(np.dot(d, e))

f = np.array([1, 2, 3, 4, 5])
print(f.shape)
g = np.array([[1, 2, 3, 4, 5]])
print(g.shape)
print(f * g)

h = np.array([[1, 2, 3], [4, 5, 6]])
print(np.max(h, axis=0))
print(np.max(h, axis=1))

i = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(i[::2])

s = "-0.3"
print(float(s))
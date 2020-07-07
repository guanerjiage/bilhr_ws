#!/usr/bin/env python

import numpy as np

a = np.zeros(20)
a[:8] = 1
np.random.shuffle(a)
b = a.reshape((4, 5))
c = np.arange(20).reshape((4, 5))

print(b)
print(c)
print(np.sum(b, axis=0))
print(np.sum(b, axis=1))
print(np.all(np.sum(b, axis=0)))
print(np.all(np.sum(b, axis=1)))
print(c[b>0])
print(np.sum(a))
print(c[b>0][1::2])
print(tuple(5 for x in range(0, 3)))
print(7.0/2)

f = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
print(f)
g = np.concatenate([f.T, f.T], axis=0)
h = g * np.array([2, 3]).reshape(-1, 1)
print(g)
print(h)
print(np.arange(1, 151, 10))



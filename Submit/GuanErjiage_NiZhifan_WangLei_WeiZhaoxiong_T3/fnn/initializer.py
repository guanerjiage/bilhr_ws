#!/usr/bin/env python
# Author: Erjiage GUAN, Zhifan NI, Lei WANG, Zhaoxiong WEI

import numpy as np

class Initializer(object):
    @staticmethod
    def generate(row, col):
        raise NotImplementedError


class RandnInitializer(Initializer):
    @staticmethod
    def generate(row, col):
        return np.random.randn(row, col)

class RandInitializer(Initializer):
    @staticmethod
    def generate(row, col):
        return np.random.rand(row, col)


class ZeroInitializer(Initializer):
    @staticmethod
    def generate(row, col):
        return np.zeros((row, col))
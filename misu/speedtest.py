#!/usr/bin/env python
# coding=utf-8

import numpy as np
import timeit
import misulib as u

a=1 * u.m
b=1 * u.cm


a_arr = np.linspace(1,100,10000) * u.m
b_arr = np.linspace(1,100,10000) * u.cm

def test_instantiation():
    q = 1 * u.m

def test_sum(a, b):
    q = a + b

def test_array():
    q = np.linspace(1,100,10000) * u.m


# call timeit
print('Instantiation: {0:2.3f} ms'.format(1000*timeit.timeit("test_instantiation()", globals=globals(), number=1000)))
print('Sum: {0:2.3f} ms'.format(1000*timeit.timeit("test_sum(e, f)", globals=globals(), number=1000)))
print('Numpy array: {0:2.3f} ms'.format(1000*timeit.timeit("test_array()", globals=globals(), number=1000)))
print('Numpy array sum: {0:2.3f} ms'.format(1000*timeit.timeit("test_sum(e_arr, f_arr)", globals=globals(), number=1000)))

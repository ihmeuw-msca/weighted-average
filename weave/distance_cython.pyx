import numpy as np
from libc.math cimport abs, pow, sqrt


cpdef double[::1] absolute(double x, double[::1] y):
    cdef int n
    cdef int ii
    cdef double[::1] distance

    n = len(y)
    distance = np.empty(n, dtype=np.float64)
    for ii in range(n):
        distance[ii] = abs(x - y[ii])
    return distance


cpdef double[::1] euclidean(double[::1] x, double[:, ::1] y):
    cdef int n
    cdef int m
    cdef int ii
    cdef double inner
    cdef double[::1] distance

    n = len(y)
    m = len(x)
    distance = np.empty(n, dtype=np.float64)
    for ii in range(n):
        inner = 0
        for jj in range(m):
            inner += pow(x[jj] - y[ii, jj], 2)
        distance[ii] = sqrt(inner)
    return distance


cpdef double[::1] hierarchical(double[::1] x, double[:, ::1] y):
    cdef int n
    cdef int ii
    cdef int jj
    cdef double[::1] distance

    n = len(y)
    distance = np.empty(n, dtype=np.float64)
    for ii in range(n):
        for jj in range(n, 0, -1):
            if x[jj - 1] == y[ii][jj - 1]:
                distance[ii] = n - ii
                break
    return distance


cpdef double[::1] dictionary(double x, double[::1] y, dict distance_dict):
    cdef int n
    cdef int ii
    cdef double[::1] distance

    n = len(y)
    distance = np.empty(n, dtype=np.float64)
    for ii in range(n):
        if x <= y[ii]:
            distance[ii] = distance_dict[(x, y)]
        else:
            distance[ii] = distance_dict[(y, x)]
    return distance

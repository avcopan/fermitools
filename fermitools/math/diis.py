import numpy
from itertools import starmap
from itertools import combinations_with_replacement as combos


def extrapolate(cs, rs):
    n = len(cs)
    # build A-matrix
    a = -numpy.ones((n+1, n+1))
    a[n, n] = 0
    a[numpy.triu_indices(n)] = list(starmap(numpy.vdot, combos(rs, r=2)))
    a.T[numpy.triu_indices(n, 1)] = a[numpy.triu_indices(n, 1)]
    # build b-vector
    b = numpy.zeros((n+1,))
    b[n] = -1
    x = numpy.linalg.solve(a, b)[:n]
    c = numpy.tensordot(x, cs, axes=(0, 0))
    return c

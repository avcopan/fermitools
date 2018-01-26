import numpy
from functools import partial
from itertools import starmap
from itertools import combinations_with_replacement as combos


def extrapolate(cs, rs):
    n = len(cs)
    # build A-matrix
    a = -numpy.ones((n+1, n+1))
    a[n, n] = 0
    a[numpy.triu_indices(n)] = list(starmap(vdot, combos(rs, r=2)))
    a.T[numpy.triu_indices(n, 1)] = a[numpy.triu_indices(n, 1)]
    # build b-vector
    b = numpy.zeros((n+1,))
    b[n] = -1
    x = numpy.linalg.solve(a, b)[:n]
    return transform_columns(cs, x)


def vdot(a, b):
    return (numpy.vdot(a, b) if isinstance(a, numpy.ndarray) else
            sum(starmap(numpy.vdot, zip(a, b))))


def transform_columns(a, m):
    trans = partial(numpy.tensordot, b=m, axes=(0, 0))
    return (trans(a) if isinstance(a[0], numpy.ndarray) else
            list(map(trans, zip(*a))))

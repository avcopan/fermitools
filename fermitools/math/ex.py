import numpy
from itertools import takewhile
from itertools import accumulate


def expm(a):
    n, m = numpy.shape(a)
    assert n == m
    factors = (a/k for k in numpy.arange(1, 100, dtype=numpy.float64))
    powers = takewhile(_too_big, accumulate(factors, func=numpy.dot))
    return numpy.eye(n) + sum(powers)


def _too_big(a):
    return numpy.amax(numpy.abs(a)) > numpy.finfo(float).eps

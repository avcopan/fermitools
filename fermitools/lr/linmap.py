import numpy


def zero(x):
    return 0.


def eye(x):
    return x


def negative(f):

    def _minus_f(x):
        return numpy.negative(f(x))

    return _minus_f


def add(f, g):

    def _f_plus_g(x):
        return numpy.add(f(x), g(x))

    return _f_plus_g


def subtract(f, g):

    def _f_minus_g(x):
        return numpy.subtract(f(x), g(x))

    return _f_minus_g


def diagonal(f, dim):

    fd = numpy.zeros(dim)
    e = numpy.zeros(dim)

    for i in range(dim):
        e[i] = 1.
        fd[i] = numpy.dot(e, f(e))
        e[i] = 0.

    return fd


def block_diag(fs, indices_or_sections):

    def _bdiag(x):
        xs = numpy.split(x, indices_or_sections)
        fxs = [f(x) for f, x in zip(fs, xs)]
        return numpy.concatenate(fxs, axis=0)

    return _bdiag


def bmat(fmatrix, indices_or_sections):

    def _bmat(x):
        xs = numpy.split(x, indices_or_sections)
        fxs = [sum(f(x) for f, x in zip(frow, xs)) for frow in fmatrix]
        return numpy.concatenate(fxs, axis=0)

    return _bmat

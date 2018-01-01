import numpy


def eye(x):
    return x


def negative(f):

    def _neg(x):
        return numpy.negative(f(x))

    return _neg


def diag(a, dim):
    """diagonal matrix elements of a linear operator

    :param a: linear operator
    :type a: typing.Callable
    :param dim: the dimension of a (if rectangular, use the smaller one)
    :type dim: int
    """
    a_diag = tuple(a(ei)[i] for i, ei in enumerate(numpy.eye(dim)))
    return numpy.array(a_diag)


def bmat(fmatrix, indices_or_sections):

    def _bmat(x):
        xs = numpy.split(x, indices_or_sections, axis=0)
        fxs = [sum(f(x) for f, x in zip(frow, xs)) for frow in fmatrix]
        return numpy.concatenate(fxs, axis=0)

    return _bmat


def block_diag(fvec, indices_or_sections):

    def _bdiag(x):
        xs = numpy.split(x, indices_or_sections, axis=0)
        fxs = [f(x) for f, x in zip(fvec, xs)]
        return numpy.concatenate(fxs, axis=0)

    return _bdiag

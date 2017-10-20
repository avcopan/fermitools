import numpy
import scipy.misc
import numbers


def central_difference(f, x, step=0.005, nder=1, npts=None):
    """differentiate a function using central differences

    :param f: the function
    :type f: typing.Callable
    :param x: the point at which to evaluate the derivative
    :type x: float or tuple
    :param step: the step size
    :type step: float
    :param nder: return the nth derivative
    :type nder: int
    :param npts: the number of grid points, default `nder` + `1` + `nder % 2`
    :type npts: int
    """
    if npts is None:
        npts = nder + 1 + nder % 2
    weights = scipy.misc.central_diff_weights(Np=npts, ndiv=nder)
    spacings = [(k - npts//2) for k in range(npts)]
    ndim = 1 if isinstance(x, numbers.Number) else len(x)

    def derivative(axis):
        dx = step * numpy.eye(ndim)[axis]
        grid = numpy.array(x) + numpy.outer(spacings, dx)
        vals = tuple(map(f, grid))
        return numpy.vdot(weights, vals) / step ** nder

    diff = numpy.array(tuple(map(derivative, range(ndim))))
    return diff[0] if isinstance(x, numbers.Number) else diff

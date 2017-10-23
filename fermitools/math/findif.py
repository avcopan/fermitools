import numpy
import scipy.misc


def central_difference(f, x, step=0.005, nder=1, npts=None):
    """differentiate a function using central differences

    :param f: the function
    :type f: typing.Callable
    :param x: the point at which to evaluate the derivative
    :type x: float or numpy.ndarrray
    :param step: step size, or a grid of step sizes corresponding to `x`
    :type step: float or numpy.ndarray
    :param nder: return the nth derivative
    :type nder: int
    :param npts: the number of grid points, default `nder` + `1` + `nder % 2`
    :type npts: int
    """
    if npts is None:
        npts = nder + 1 + nder % 2
    if numpy.ndim(step) == 0:
        step = float(step) * numpy.ones_like(x)

    weights = scipy.misc.central_diff_weights(Np=npts, ndiv=nder)

    def derivative(index):
        dx = numpy.zeros_like(x)
        dx[index] = step[index]
        grid = [numpy.array(x) + (k - npts//2) * dx for k in range(npts)]
        vals = tuple(map(f, grid))
        return numpy.vdot(weights, vals) / (step[index] ** nder)

    der = tuple(map(derivative, numpy.ndindex(numpy.shape(x))))
    return numpy.reshape(der, numpy.shape(x))

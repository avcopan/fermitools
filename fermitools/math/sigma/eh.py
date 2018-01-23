import numpy
import scipy.linalg
import warnings
import sys

from ..ot import orth


def eighg(a, b, neig, ad, bd, guess, niter=100, nvec=100, r_thresh=1e-5,
          print_conv=True, highest=False):
    """solve for the lowest generalized eigenvalues of a hermitian matrix

    :param a: the matrix, as a callable linear operator
    :type a: typing.Callable
    :param b: the metric, as a callable linear operator
    :type b: typing.Callable
    :param neig: the number of eigenvalues to solve
    :type neig: int
    :param ad: the diagonal elements of a, or an approximation to them
    :type ad: numpy.ndarray
    :param bd: the diagonal elements of b, or an approximation to them
    :type bd: numpy.ndarray
    :param guess: initial guess vectors
    :type guess: numpy.ndarray
    :param niter: the maximum number of iterations
    :type niter: int
    :param nvec: the maximum number of vectors to hold in memory
    :type nvec: int
    :param r_thresh: residual convergence threshold
    :type r_thresh: float
    :param print_conv: print convergence info?
    :type print_conv: bool
    :param highest: compute the highest roots, instead of the lowest ones?
    :type highest: bool

    :returns: eigenvalues, eigenvectors, convergence info
    :rtype: (numpy.ndarray, numpy.ndarray, dict)
    """
    dim, _ = guess.shape

    v1 = guess
    av = bv = v = numpy.zeros((dim, 0))

    slc = slice(None, neig) if not highest else slice(None, -neig-1, -1)

    for iteration in range(niter):
        v = numpy.concatenate((v, v1), axis=1)
        av = numpy.concatenate((av, a(v1)), axis=1)
        bv = numpy.concatenate((bv, b(v1)), axis=1)
        _, rdim = v.shape

        a_red = numpy.dot(v.T, av)
        b_red = numpy.dot(v.T, bv)

        vals, vecs = scipy.linalg.eigh(a=a_red, b=b_red)

        w = vals[slc]
        u = vecs[:, slc]

        x = numpy.dot(v, u)
        ax = numpy.dot(av, u)
        bx = numpy.dot(bv, u)

        r = ax - bx * w
        r_max = numpy.amax(numpy.abs(r))

        info = {'niter': iteration + 1, 'rdim': rdim, 'r_max': r_max}

        converged = r_max < r_thresh

        if print_conv:
            print(info)
            # (TEMPORARY HACK -- DELETE THIS LATER)
            print(1/w)
            sys.stdout.flush()

        if converged:
            break

        denom = numpy.reshape(w[None, :] * bd[:, None] - ad[:, None], r.shape)
        vstep = r / denom
        v1 = orth(vstep, against=v)
        _, rdim1 = v1.shape

        if rdim + rdim1 > nvec:
            av = bv = v = numpy.zeros((dim, 0))
            v1 = x

    if not converged:
        warnings.warn("Did not converge! (r_max: {:7.1e})".format(r_max))

    return w, x, info

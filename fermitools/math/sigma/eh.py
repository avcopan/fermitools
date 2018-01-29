import numpy
import scipy.linalg
import warnings
import sys

from ..ot import orth


def eighg(a, b, neig, ad, bd, guess, niter=100, nsvec=100, nvec=100,
          rthresh=1e-5, print_conv=True, highest=False):
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
    :param nsvec: maximum number of sigma vectors to compute per sub-iteration
    :type nsvec: int
    :param nvec: maximum number of vectors to hold in memory
    :type nvec: int
    :param rthresh: residual convergence threshold
    :type rthresh: float
    :param print_conv: print convergence info?
    :type print_conv: bool
    :param highest: compute the highest roots, instead of the lowest ones?
    :type highest: bool

    :returns: eigenvalues, eigenvectors, convergence info
    :rtype: (numpy.ndarray, numpy.ndarray, dict)
    """
    dim, _ = guess.shape

    vnew = guess
    av = bv = v = numpy.zeros((dim, 0))

    slc = slice(None, neig) if not highest else slice(None, -neig-1, -1)

    for iteration in range(niter):
        _, nnew = vnew.shape
        sections = numpy.arange(nsvec, nnew, nsvec)
        for i, vi in enumerate(numpy.split(vnew, sections, axis=1)):
            av = numpy.concatenate((av, a(vi)), axis=1)
            bv = numpy.concatenate((bv, b(vi)), axis=1)
            _, rdim = numpy.shape(av)
            print('subiteration {:d}, rdim={:d}'.format(i, rdim))

        v = numpy.concatenate((v, vnew), axis=1)
        a_red = numpy.dot(v.T, av)
        b_red = numpy.dot(v.T, bv)

        vals, vecs = scipy.linalg.eigh(a=a_red, b=b_red)

        w = vals[slc]
        x_red = vecs[:, slc]

        x = numpy.dot(v, x_red)
        ax = numpy.dot(av, x_red)
        bx = numpy.dot(bv, x_red)

        r = ax - bx * w
        rmax = numpy.amax(numpy.abs(r))

        info = {'niter': iteration + 1, 'rdim': rdim, 'rmax': rmax}

        converged = rmax < rthresh

        if print_conv:
            print(info)
            # (TEMPORARY HACK -- DELETE THIS LATER)
            print(1/w)
            sys.stdout.flush()

        if converged:
            break

        denom = numpy.reshape(w[None, :] * bd[:, None] - ad[:, None], r.shape)
        vstep = r / denom
        vnew = orth(vstep, against=v)
        _, rdim1 = vnew.shape

        if rdim + rdim1 > nvec:
            av = bv = v = numpy.zeros((dim, 0))
            vnew = x

    if not converged:
        warnings.warn("Did not converge! (rmax: {:7.1e})".format(rmax))

    return w, x, info

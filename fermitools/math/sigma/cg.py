import numpy
import scipy.linalg
import warnings
import sys

from ..ot import orth


def solve(a, b, ad, guess, niter=100, nvec=100, rthresh=1e-5, print_conv=True):
    dim, _ = guess.shape

    vnew = guess
    av = v = numpy.zeros((dim, 0))

    for iteration in range(niter):
        v = numpy.concatenate((v, vnew), axis=1)
        av = numpy.concatenate((av, a(vnew)), axis=1)
        _, rdim = v.shape

        a_red = numpy.dot(v.T, av)
        b_red = numpy.dot(v.T, b)
        x_red = scipy.linalg.solve(a=a_red, b=b_red)

        x = numpy.dot(v, x_red)
        ax = numpy.dot(av, x_red)
        r = ax - b
        rmax = numpy.amax(numpy.abs(r))

        info = {'niter': iteration + 1, 'rdim': rdim, 'rmax': rmax}

        converged = rmax < rthresh

        if print_conv:
            print(info)
            sys.stdout.flush()

        if converged:
            break

        denom = -ad[:, None] if numpy.ndim(r) == 2 else -ad
        vstep = r / denom
        vnew = orth(vstep, against=v)
        _, rdim1 = vnew.shape

        if rdim + rdim1 > nvec:
            av = v = numpy.zeros((dim, 0))
            vnew = x

    if not converged:
        warnings.warn("Did not converge! (rmax: {:7.1e})".format(rmax))

    return x, info

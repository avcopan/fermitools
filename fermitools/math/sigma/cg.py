import numpy
import scipy.linalg
import warnings
import sys

from ..ot import orth


def solve(a, b, ad, guess, niter=100, nvec=100, r_thresh=1e-5,
          print_conv=True):
    dim, _ = guess.shape

    v1 = guess
    av = v = numpy.zeros((dim, 0))

    for iteration in range(niter):
        v = numpy.concatenate((v, v1), axis=1)
        av = numpy.concatenate((av, a(v1)), axis=1)
        _, rdim = v.shape

        a_red = numpy.dot(v.T, av)
        b_red = numpy.dot(v.T, b)
        x_red = scipy.linalg.solve(a=a_red, b=b_red)

        x = numpy.dot(v, x_red)
        ax = numpy.dot(av, x_red)
        r = ax - b
        r_max = numpy.amax(numpy.abs(r))

        info = {'niter': iteration + 1, 'rdim': rdim, 'r_max': r_max}

        converged = r_max < r_thresh

        if print_conv:
            print(info)
            sys.stdout.flush()

        if converged:
            break

        denom = -ad[:, None] if numpy.ndim(r) == 2 else -ad
        vstep = r / denom
        v1 = orth(vstep, against=v)
        _, rdim1 = v1.shape

        if rdim + rdim1 > nvec:
            av = v = numpy.zeros((dim, 0))
            v1 = x

    if not converged:
        warnings.warn("Did not converge! (r_max: {:7.1e})".format(r_max))

    return x, info

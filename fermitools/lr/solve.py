import numpy
from ..math.sigma import add
from ..math.sigma import bmat
from ..math.sigma import negative
from ..math.sigma import eighg
from ..math.sigma import solve


def static_response(a, b, pg, ad, nvec=100, niter=50, rthresh=1e-5):
    e = add(a, b)
    v = -2*pg
    guess = v / ad[:, None]

    r, info = solve(
            a=e, b=v, ad=ad, guess=guess, niter=niter, nvec=nvec,
            rthresh=rthresh)

    return r, info


def spectrum(a, b, s, d, ad, sd, nroot=1, nguess=10, nsvec=10, nvec=100,
             niter=50, rthresh=1e-7, guess_type=None, disk=False):
    e = bmat([[a, b], [b, a]], 2)
    m = bmat([[s, d], [negative(d), negative(s)]], 2)
    ed = numpy.concatenate((+ad, +ad))
    md = numpy.concatenate((+sd, -sd))
    w_inv, z, info = eighg(
            a=m, b=e, neig=nroot, ad=md, bd=ed, nguess=nguess*nroot,
            rthresh=rthresh, nsvec=nsvec, nvec=nvec*nroot, niter=niter,
            highest=True, guess_type=guess_type, disk=disk)
    w = 1. / w_inv
    x, y = numpy.split(z, 2)
    return w, x, y, info

import numpy
import scipy
from ..math.sigma import bmat
from ..math.sigma import negative
from ..math.sigma import evec_guess
from ..math.sigma import eighg


def static_response(a, b, pg):
    a_ = a(numpy.eye(len(pg)))
    b_ = b(numpy.eye(len(pg)))

    e = a_ + b_
    r = scipy.linalg.solve(e, -2*pg)
    return r


def spectrum(a, b, s, d, ad, sd, nroot=1, nguess=None, nvec=None, niter=50,
             r_thresh=1e-6):
    nguess = 2 if nguess is None else nguess
    nvec = 2 if nvec is None else nvec

    e = bmat([[a, b], [b, a]], 2)
    m = bmat([[s, d], [negative(d), negative(s)]], 2)
    ed = numpy.concatenate((+ad, +ad))
    md = numpy.concatenate((+sd, -sd))

    guess = evec_guess(md, nguess*nroot, bd=ed, highest=True)
    v, u, info = eighg(
            a=m, b=e, neig=nroot, ad=md, bd=ed, guess=guess,
            r_thresh=r_thresh, nvec=nvec*nroot, niter=niter, highest=True)
    w = 1. / v

    return w, u, info

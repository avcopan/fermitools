import numpy
import scipy
import scipy.sparse.linalg
import warnings
import itertools
import functools

from ..math.sigma import diag
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


def static_response_new(a, b, pg, ad):
    n = len(pg)
    a_ = scipy.sparse.linalg.LinearOperator((n, n), matvec=a)
    b_ = scipy.sparse.linalg.LinearOperator((n, n), matvec=b)
    pc = diag(1./ad)
    pc_ = scipy.sparse.linalg.LinearOperator((n, n), matvec=pc)
    v = -2*pg

    e_ = a_ + b_

    r, info = scipy.sparse.linalg.cg(A=e_, b=v, x0=pc_(v), M=pc_)
    r_solver_ = functools.partial(scipy.sparse.linalg.cg, e_, M=pc_)
    rs, infos = zip(*itertools.starmap(r_solver_, zip(v.T, pc_(v).T)))
    r = numpy.moveaxis(tuple(rs), -1, 0)

    if any(info != 0 for info in infos):
        warnings.warn("Conjugate gradient solver did not converge!")

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

import numpy
import scipy
import functools
from toolz import functoolz

import fermitools


def a_sigma(a11, a12, a21, a22):

    def _a(r):
        r1, r2 = r
        return (a11(r1) + a12(r2), a21(r1) + a22(r2))

    return _a


def solve_spectrum(nroots, nocc, norb, a11, b11, a12, b12, a21, b21, a22, b22,
                   x11):
    xa11 = functoolz.compose(x11, a11)
    xb11 = functoolz.compose(x11, b11)
    xa12 = functoolz.compose(x11, a12)
    xb12 = functoolz.compose(x11, b12)

    ea_ = a_sigma(fermitools.func.add(xa11, xb11),
                  fermitools.func.add(xa12, xb12),
                  fermitools.func.add(a21, b21),
                  fermitools.func.add(a22, b22))

    es_ = a_sigma(fermitools.func.sub(xa11, xb11),
                  fermitools.func.sub(xa12, xb12),
                  fermitools.func.sub(a21, b21),
                  fermitools.func.sub(a22, b22))

    no, nv = nocc, norb-nocc
    ns = no * nv
    nd = no * (no - 1) * nv * (nv - 1) // 4
    split = splitter(ns)
    bmat_unravel = bmat_unraveler(no, nv)
    e_matvec = functoolz.compose(
            join, bmat_ravel, ea_, es_, bmat_unravel, split)
    e_eff_ = scipy.sparse.linalg.LinearOperator(
            shape=(ns+nd, ns+nd), matvec=e_matvec)

    w2, u = scipy.sparse.linalg.eigs(e_eff_, k=nroots, which='SR')
    w = numpy.sqrt(numpy.real(w2))
    sortv = numpy.argsort(w2)
    return w[sortv], u[:, sortv]


def solve_static_response(nocc, norb, a11, b11, a12, b12, a21, b21, a22, b22,
                          pg1, pg2):

    ea_ = a_sigma(fermitools.func.add(a11, b11),
                  fermitools.func.add(a12, b12),
                  fermitools.func.add(a21, b21),
                  fermitools.func.add(a22, b22))

    no, nv = nocc, norb-nocc
    ns = no * nv
    nd = no * (no - 1) * nv * (nv - 1) // 4
    split = splitter(ns)
    bmat_unravel = bmat_unraveler(no, nv)
    e_matvec = functoolz.compose(join, bmat_ravel, ea_, bmat_unravel, split)
    e_eff_ = scipy.sparse.linalg.LinearOperator(
            shape=(ns+nd, ns+nd), matvec=e_matvec)
    pg = join(bmat_ravel((pg1, pg2)))

    r_solver_ = functools.partial(scipy.sparse.linalg.cg, e_eff_)
    rs, _ = zip(*map(r_solver_, -2 * numpy.moveaxis(pg, -1, 0)))
    r = numpy.moveaxis(tuple(rs), -1, 0)
    return numpy.tensordot(r, pg, axes=(0, 0))


# Helpers
def splitter(ns):

    def _split(r):
        r1, r2 = numpy.split(r, (ns,))
        return (r1, r2)

    return _split


def join(r):
    r1, r2 = r
    return numpy.concatenate((r1, r2), axis=0)


def bmat_unraveler(no, nv):

    def _unravel(r):
        rr1, rr2 = r
        r1 = fermitools.math.unravel(rr1, {0: {0: no, 1: nv}})
        r2 = fermitools.math.asym.megaunravel(
                rr2, {0: {(0, 1): no, (2, 3): nv}})
        return (r1, r2)

    return _unravel


def bmat_ravel(r):
    ur1, ur2 = r
    r1 = fermitools.math.ravel(ur1, {0: (0, 1)})
    r2 = fermitools.math.asym.megaravel(ur2, {0: ((0, 1), (2, 3))})
    return (r1, r2)

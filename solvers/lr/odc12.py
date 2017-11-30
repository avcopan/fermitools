import numpy
import scipy
import functools
from toolz import functoolz

import fermitools


def a_(a_d1d1_, a_d1d2_, a_d2d1_, a_d2d2_):

    def _sigma(r):
        r1, r2 = r
        return (a_d1d1_(r1) + a_d1d2_(r2), a_d2d1_(r1) + a_d2d2_(r2))

    return _sigma


def solve_spectrum(nroots, nocc, norb, a_d1d1_, b_d1d1_, a_d1d2_, b_d1d2_,
                   a_d2d1_, b_d2d1_, a_d2d2_, b_d2d2_, x1_):
    a_x1d1_ = functoolz.compose(x1_, a_d1d1_)
    b_x1d1_ = functoolz.compose(x1_, b_d1d1_)
    a_x1d2_ = functoolz.compose(x1_, a_d1d2_)
    b_x1d2_ = functoolz.compose(x1_, b_d1d2_)

    ea_ = a_(fermitools.func.add(a_x1d1_, b_x1d1_),
             fermitools.func.add(a_x1d2_, b_x1d2_),
             fermitools.func.add(a_d2d1_, b_d2d1_),
             fermitools.func.add(a_d2d2_, b_d2d2_))
    es_ = a_(fermitools.func.sub(a_x1d1_, b_x1d1_),
             fermitools.func.sub(a_x1d2_, b_x1d2_),
             fermitools.func.sub(a_d2d1_, b_d2d1_),
             fermitools.func.sub(a_d2d2_, b_d2d2_))

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


def solve_static_response(nocc, norb, a_d1d1_, b_d1d1_, a_d1d2_, b_d1d2_,
                          a_d2d1_, b_d2d1_, a_d2d2_, b_d2d2_, t_d1, t_d2):
    ea_ = a_(fermitools.func.add(a_d1d1_, b_d1d1_),
             fermitools.func.add(a_d1d2_, b_d1d2_),
             fermitools.func.add(a_d2d1_, b_d2d1_),
             fermitools.func.add(a_d2d2_, b_d2d2_))

    no, nv = nocc, norb-nocc
    ns = no * nv
    nd = no * (no - 1) * nv * (nv - 1) // 4
    split = splitter(ns)
    bmat_unravel = bmat_unraveler(no, nv)
    e_matvec = functoolz.compose(join, bmat_ravel, ea_, bmat_unravel, split)
    e_eff_ = scipy.sparse.linalg.LinearOperator(
            shape=(ns+nd, ns+nd), matvec=e_matvec)
    t = join(bmat_ravel((t_d1, t_d2)))

    r_solver_ = functools.partial(scipy.sparse.linalg.cg, e_eff_)
    rs, _ = zip(*map(r_solver_, -2 * numpy.moveaxis(t, -1, 0)))
    r = numpy.moveaxis(tuple(rs), -1, 0)
    return numpy.tensordot(r, t, axes=(0, 0))


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

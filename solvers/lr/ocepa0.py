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


def solve_spectrum(norb, nocc, h_aso, g_aso, c, t2, nroots=1):
    co, cv = numpy.split(c, (nocc,), axis=1)
    hoo = fermitools.math.transform(h_aso, {0: co, 1: co})
    hov = fermitools.math.transform(h_aso, {0: co, 1: cv})
    hvv = fermitools.math.transform(h_aso, {0: cv, 1: cv})
    goooo = fermitools.math.transform(g_aso, {0: co, 1: co, 2: co, 3: co})
    gooov = fermitools.math.transform(g_aso, {0: co, 1: co, 2: co, 3: cv})
    goovv = fermitools.math.transform(g_aso, {0: co, 1: co, 2: cv, 3: cv})
    govov = fermitools.math.transform(g_aso, {0: co, 1: cv, 2: co, 3: cv})
    govvv = fermitools.math.transform(g_aso, {0: co, 1: cv, 2: cv, 3: cv})
    gvvvv = fermitools.math.transform(g_aso, {0: cv, 1: cv, 2: cv, 3: cv})
    dm1oo = numpy.eye(nocc)
    cm1oo, cm1vv = fermitools.oo.ocepa0.onebody_correlation_density(t2)
    m1oo = dm1oo + cm1oo
    m1vv = cm1vv
    k2oooo = fermitools.oo.ocepa0.twobody_cumulant_oooo(t2)
    k2oovv = fermitools.oo.ocepa0.twobody_cumulant_oovv(t2)
    k2ovov = fermitools.oo.ocepa0.twobody_cumulant_ovov(t2)
    k2vvvv = fermitools.oo.ocepa0.twobody_cumulant_vvvv(t2)

    m2oooo = fermitools.oo.ocepa0.twobody_moment_oooo(dm1oo, cm1oo, k2oooo)
    m2oovv = fermitools.oo.ocepa0.twobody_moment_oovv(k2oovv)
    m2ovov = fermitools.oo.ocepa0.twobody_moment_ovov(dm1oo, cm1vv, k2ovov)
    m2vvvv = fermitools.oo.ocepa0.twobody_moment_vvvv(k2vvvv)

    foo = fermitools.oo.ocepa0.fock_oo(hoo, goooo)
    fov = fermitools.oo.ocepa0.fock_oo(hov, gooov)
    fvv = fermitools.oo.ocepa0.fock_vv(hvv, govov)

    no, nv = nocc, norb-nocc
    sinv1 = scipy.linalg.inv(fermitools.lr.ocepa0.s1_matrix(m1oo, m1vv))
    sinv_d1d1 = fermitools.math.unravel(
            sinv1, {0: {0: no, 1: nv}, 1: {2: no, 3: nv}})

    a_d1d1_ = fermitools.lr.ocepa0.a_d1d1_(
           hoo, hvv, goooo, goovv, govov, gvvvv, m1oo, m1vv, m2oooo, m2oovv,
           m2ovov, m2vvvv, sinv1=sinv_d1d1)
    b_d1d1_ = fermitools.lr.ocepa0.b_d1d1_(
           goooo, goovv, govov, gvvvv, m2oooo, m2oovv, m2ovov, m2vvvv,
           sinv1=sinv_d1d1)
    a_d1d2_ = fermitools.lr.ocepa0.a_d1d2_(fov, gooov, govvv, t2,
                                           sinv1=sinv_d1d1)
    b_d1d2_ = fermitools.lr.ocepa0.b_d1d2_(fov, gooov, govvv, t2,
                                           sinv1=sinv_d1d1)
    a_d2d1_ = fermitools.lr.ocepa0.a_d2d1_(fov, gooov, govvv, t2)
    b_d2d1_ = fermitools.lr.ocepa0.b_d2d1_(fov, gooov, govvv, t2)
    a_d2d2_ = fermitools.lr.ocepa0.a_d2d2_(foo, fvv, goooo, govov, gvvvv)

    ea_ = a_(fermitools.func.add(a_d1d1_, b_d1d1_),
             fermitools.func.add(a_d1d2_, b_d1d2_),
             fermitools.func.add(a_d2d1_, b_d2d1_), a_d2d2_)
    es_ = a_(fermitools.func.sub(a_d1d1_, b_d1d1_),
             fermitools.func.sub(a_d1d2_, b_d1d2_),
             fermitools.func.sub(a_d2d1_, b_d2d1_), a_d2d2_)

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


def solve_static_response(norb, nocc, h_aso, p_aso, g_aso, c, t2):
    co, cv = numpy.split(c, (nocc,), axis=1)
    hoo = fermitools.math.transform(h_aso, {0: co, 1: co})
    hov = fermitools.math.transform(h_aso, {0: co, 1: cv})
    hvv = fermitools.math.transform(h_aso, {0: cv, 1: cv})
    poo = fermitools.math.transform(p_aso, {1: co, 2: co})
    pov = fermitools.math.transform(p_aso, {1: co, 2: cv})
    pvv = fermitools.math.transform(p_aso, {1: cv, 2: cv})
    goooo = fermitools.math.transform(g_aso, {0: co, 1: co, 2: co, 3: co})
    gooov = fermitools.math.transform(g_aso, {0: co, 1: co, 2: co, 3: cv})
    goovv = fermitools.math.transform(g_aso, {0: co, 1: co, 2: cv, 3: cv})
    govov = fermitools.math.transform(g_aso, {0: co, 1: cv, 2: co, 3: cv})
    govvv = fermitools.math.transform(g_aso, {0: co, 1: cv, 2: cv, 3: cv})
    gvvvv = fermitools.math.transform(g_aso, {0: cv, 1: cv, 2: cv, 3: cv})
    dm1oo = numpy.eye(nocc)
    cm1oo, cm1vv = fermitools.oo.ocepa0.onebody_correlation_density(t2)
    m1oo = dm1oo + cm1oo
    m1vv = cm1vv
    k2oooo = fermitools.oo.ocepa0.twobody_cumulant_oooo(t2)
    k2oovv = fermitools.oo.ocepa0.twobody_cumulant_oovv(t2)
    k2ovov = fermitools.oo.ocepa0.twobody_cumulant_ovov(t2)
    k2vvvv = fermitools.oo.ocepa0.twobody_cumulant_vvvv(t2)

    m2oooo = fermitools.oo.ocepa0.twobody_moment_oooo(dm1oo, cm1oo, k2oooo)
    m2oovv = fermitools.oo.ocepa0.twobody_moment_oovv(k2oovv)
    m2ovov = fermitools.oo.ocepa0.twobody_moment_ovov(dm1oo, cm1vv, k2ovov)
    m2vvvv = fermitools.oo.ocepa0.twobody_moment_vvvv(k2vvvv)

    foo = fermitools.oo.ocepa0.fock_oo(hoo, goooo)
    fov = fermitools.oo.ocepa0.fock_oo(hov, gooov)
    fvv = fermitools.oo.ocepa0.fock_vv(hvv, govov)

    t_d1 = fermitools.lr.ocepa0.t_d1(pov, m1oo, m1vv)
    t_d2 = fermitools.lr.ocepa0.t_d2(poo, pvv, t2)
    a_d1d1_ = fermitools.lr.ocepa0.a_d1d1_(
           hoo, hvv, goooo, goovv, govov, gvvvv, m1oo, m1vv, m2oooo, m2oovv,
           m2ovov, m2vvvv)
    b_d1d1_ = fermitools.lr.ocepa0.b_d1d1_(
           goooo, goovv, govov, gvvvv, m2oooo, m2oovv, m2ovov, m2vvvv,)
    a_d1d2_ = fermitools.lr.ocepa0.a_d1d2_(fov, gooov, govvv, t2)
    b_d1d2_ = fermitools.lr.ocepa0.b_d1d2_(fov, gooov, govvv, t2)
    a_d2d1_ = fermitools.lr.ocepa0.a_d2d1_(fov, gooov, govvv, t2)
    b_d2d1_ = fermitools.lr.ocepa0.b_d2d1_(fov, gooov, govvv, t2)
    a_d2d2_ = fermitools.lr.ocepa0.a_d2d2_(foo, fvv, goooo, govov, gvvvv)

    ea_ = a_(fermitools.func.add(a_d1d1_, b_d1d1_),
             fermitools.func.add(a_d1d2_, b_d1d2_),
             fermitools.func.add(a_d2d1_, b_d2d1_), a_d2d2_)

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

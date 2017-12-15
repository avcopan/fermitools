import numpy
import scipy
import time

import fermitools
import solvers
import interfaces.psi4 as interface

import os
from numpy.testing import assert_almost_equal

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         '../data')

CHARGE = +0
SPIN = 0
BASIS = 'sto-3g'
LABELS = ('O', 'H', 'H')
COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
          (0.000000000000,  1.638036840407,  1.136548822547),
          (0.000000000000, -1.638036840407,  1.136548822547))
W_REF = numpy.load(os.path.join(data_path, 'neutral/odc12/w.npy'))


def main():
    # Spaces
    na = fermitools.chem.elec.count_alpha(LABELS, CHARGE, SPIN)
    nb = fermitools.chem.elec.count_beta(LABELS, CHARGE, SPIN)
    nocc = na + nb

    # Integrals
    nbf = interface.integrals.nbf(BASIS, LABELS)
    norb = 2 * nbf
    h_ao = interface.integrals.core_hamiltonian(BASIS, LABELS, COORDS)
    r_ao = interface.integrals.repulsion(BASIS, LABELS, COORDS)

    h_aso = fermitools.math.spinorb.expand(h_ao, brakets=((0, 1),))
    r_aso = fermitools.math.spinorb.expand(r_ao, brakets=((0, 2), (1, 3)))
    g_aso = r_aso - numpy.transpose(r_aso, (0, 1, 3, 2))

    # Orbitals
    ac, bc = interface.hf.unrestricted_orbitals(
            BASIS, LABELS, COORDS, CHARGE, SPIN)
    c_unsrt = scipy.linalg.block_diag(ac, bc)
    sortvec = fermitools.math.spinorb.ab2ov(dim=nbf, na=na, nb=nb)
    c_unsrt = scipy.linalg.block_diag(ac, bc)
    c = fermitools.math.spinorb.sort(c_unsrt, order=sortvec, axes=(1,))

    # Solve OCEPA0
    t2_guess = numpy.zeros((nocc, nocc, norb-nocc, norb-nocc))
    en_elec, c, t2 = solvers.oo.odc12.solve(
            norb=norb, nocc=nocc, h_aso=h_aso, g_aso=g_aso, c_guess=c,
            t2_guess=t2_guess, niter=200, e_thresh=1e-14, r_thresh=1e-13,
            print_conv=True)
    en_nuc = fermitools.chem.nuc.energy(labels=LABELS, coords=COORDS)
    en_tot = en_elec + en_nuc
    print("\nGround state energy:")
    print('{:20.15f}'.format(en_tot))

    # Define LR inputs
    no, nv = nocc, norb-nocc
    ns = no * nv
    nd = no * (no - 1) * nv * (nv - 1) // 4
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
    cm1oo, m1vv = fermitools.oo.odc12.onebody_correlation_density(t2)
    m1oo = numpy.eye(nocc) + cm1oo
    k2oooo = fermitools.oo.odc12.twobody_cumulant_oooo(t2)
    k2oovv = fermitools.oo.odc12.twobody_cumulant_oovv(t2)
    k2ovov = fermitools.oo.odc12.twobody_cumulant_ovov(t2)
    k2vvvv = fermitools.oo.odc12.twobody_cumulant_vvvv(t2)

    m2oooo = fermitools.oo.odc12.twobody_moment_oooo(m1oo, k2oooo)
    m2oovv = fermitools.oo.odc12.twobody_moment_oovv(k2oovv)
    m2ovov = fermitools.oo.odc12.twobody_moment_ovov(m1oo, m1vv, k2ovov)
    m2vvvv = fermitools.oo.odc12.twobody_moment_vvvv(m1vv, k2vvvv)

    foo = fermitools.oo.odc12.fock_oo(hoo, goooo, govov, m1oo, m1vv)
    fov = fermitools.oo.odc12.fock_oo(hov, gooov, govvv, m1oo, m1vv)
    fvv = fermitools.oo.odc12.fock_vv(hvv, govov, gvvvv, m1oo, m1vv)
    ffoo = fermitools.oo.odc12.fancy_property(foo, m1oo)
    ffvv = fermitools.oo.odc12.fancy_property(fvv, m1vv)
    fioo, fivv = fermitools.lr.odc12.fancy_mixed_interaction(
            fov, gooov, govvv, m1oo, m1vv)
    fgoooo, fgovov, fgvvvv = fermitools.lr.odc12.fancy_repulsion(
            ffoo, ffvv, goooo, govov, gvvvv, m1oo, m1vv)

    a11 = fermitools.lr.odc12.a11_sigma(
           hoo, hvv, goooo, goovv, govov, gvvvv, m1oo, m1vv, m2oooo, m2oovv,
           m2ovov, m2vvvv)
    b11 = fermitools.lr.odc12.b11_sigma(
           goooo, goovv, govov, gvvvv, m2oooo, m2oovv, m2ovov, m2vvvv)
    s11 = fermitools.lr.odc12.s11_sigma(m1oo, m1vv)
    a12 = fermitools.lr.odc12.a12_sigma(gooov, govvv, fioo, fivv, t2)
    b12 = fermitools.lr.odc12.b12_sigma(gooov, govvv, fioo, fivv, t2)
    a21 = fermitools.lr.odc12.a21_sigma(gooov, govvv, fioo, fivv, t2)
    b21 = fermitools.lr.odc12.b21_sigma(gooov, govvv, fioo, fivv, t2)
    a22 = fermitools.lr.odc12.a22_sigma(
           ffoo, ffvv, goooo, govov, gvvvv, fgoooo, fgovov, fgvvvv, t2)
    b22 = fermitools.lr.odc12.b22_sigma(fgoooo, fgovov, fgvvvv, t2)

    r1_ = fermitools.math.raveler({0: (0, 1)})
    u1_ = fermitools.math.unraveler({0: {0: no, 1: nv}})
    r2_ = fermitools.math.asym.megaraveler({0: ((0, 1), (2, 3))})
    u2_ = fermitools.math.asym.megaunraveler({0: {(0, 1): no, (2, 3): nv}})

    def combine_blocks(bl11, bl12, bl21, bl22):

        def _bl(r):
            r1r, r2r = numpy.split(r, (ns,), axis=0)
            r1u, r2u = u1_(r1r), u2_(r2r)
            sig1 = r1_(bl11(r1u) + bl12(r2u))
            sig2 = r2_(bl21(r1u) + bl22(r2u))
            return numpy.concatenate((sig1, sig2), axis=0)

        return _bl

    def nullmap(r):
        return 0.

    def idmap(r):
        return r

    print(nd)
    a = combine_blocks(bl11=a11, bl12=a12, bl21=a21, bl22=a22)
    b = combine_blocks(bl11=b11, bl12=b12, bl21=b21, bl22=b22)
    s = combine_blocks(bl11=s11, bl12=nullmap, bl21=nullmap, bl22=idmap)

    def e(r):
        ru, rl = numpy.split(r, 2, axis=0)
        sigu = a(ru) + b(rl)
        sigl = b(ru) + a(rl)
        return numpy.concatenate((sigu, sigl), axis=0)

    def m(r):
        ru, rl = numpy.split(r, 2, axis=0)
        sigu = +s(ru)
        sigl = -s(rl)
        return numpy.concatenate((sigu, sigl), axis=0)

    # Solve excitation energies
    dim = 2 * (ns + nd)
    neig = 7

    t0 = time.time()
    e_mat = e(numpy.eye(dim))
    m_mat = m(numpy.eye(dim))
    vals, vecs = scipy.linalg.eigh(m_mat, b=e_mat)
    DT = time.time() - t0
    W = -1. / vals[:neig]
    U = vecs[:, :neig]
    print("numpy:")
    print(W)
    print(U.shape)
    print(DT)
    assert_almost_equal(W, W_REF[:neig])

    nguess = neig + 3
    nvec = 50
    niter = 100
    r_thresh = 1e-7
    ed = fermitools.math.linalg.direct.diag(e, dim=dim)
    md = fermitools.math.linalg.direct.diag(m, dim=dim)

    # Perfect guess
    v, u, info = fermitools.math.linalg.direct.eighg(
            a=m, b=e, neig=neig, ad=md, bd=ed, guess=U, r_thresh=r_thresh,
            nvec=nvec, niter=niter)
    w = -1. / v
    print("perfect guess:")
    print(w)
    print(info)
    assert info['niter'] == 1
    assert info['rdim'] == neig
    assert_almost_equal(w, W, decimal=10)

    # Approximate guess
    t0 = time.time()
    guess = fermitools.math.linalg.direct.evec_guess(md, nguess, bd=ed)
    v, u, info = fermitools.math.linalg.direct.eighg(
            a=m, b=e, neig=neig, ad=md, bd=ed, guess=guess,
            r_thresh=r_thresh, nvec=nvec, niter=niter)
    w = -1. / v
    dt = time.time() - t0
    print("approximate guess:")
    print(w)
    print(info)
    print(dt)
    assert_almost_equal(w, W, decimal=10)


if __name__ == '__main__':
    main()

import numpy
import scipy
import time

import fermitools
import interfaces.psi4 as interface

import os
from numpy.testing import assert_almost_equal

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

CHARGE = +0
SPIN = 0
BASIS = 'sto-3g'
LABELS = ('O', 'H', 'H')
COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
          (0.000000000000,  1.638036840407,  1.136548822547),
          (0.000000000000, -1.638036840407,  1.136548822547))
W_REF = numpy.load(os.path.join(data_path, 'neutral/w.npy'))


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
    c_guess = fermitools.math.spinorb.sort(c_unsrt, order=sortvec, axes=(1,))

    # Solve
    t2_guess = numpy.zeros((nocc, nocc, norb-nocc, norb-nocc))
    en_elec, c, t2, info = fermitools.oo.odc12.solve(
            h_aso=h_aso, g_aso=g_aso, c_guess=c_guess, t2_guess=t2_guess,
            niter=200, r_thresh=1e-14)
    en_nuc = fermitools.chem.nuc.energy(labels=LABELS, coords=COORDS)
    en_tot = en_elec + en_nuc
    print("\nGround state energy:")
    print('{:20.15f}'.format(en_tot))

    # Define LR inputs
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
    m1oo, m1vv = fermitools.oo.odc12.onebody_density(t2)

    foo = fermitools.oo.odc12.fock_xy(
            hxy=hoo, goxoy=goooo, gxvyv=govov, m1oo=m1oo, m1vv=m1vv)
    fov = fermitools.oo.odc12.fock_xy(
            hxy=hov, goxoy=gooov, gxvyv=govvv, m1oo=m1oo, m1vv=m1vv)
    fvv = fermitools.oo.odc12.fock_xy(
            hxy=hvv, goxoy=govov, gxvyv=gvvvv, m1oo=m1oo, m1vv=m1vv)
    ffoo = fermitools.oo.odc12.fancy_property(foo, m1oo)
    ffvv = fermitools.oo.odc12.fancy_property(fvv, m1vv)

    fioo, fivv = fermitools.lr.odc12.fancy_mixed_interaction(
            fov, gooov, govvv, m1oo, m1vv)
    fgoooo, fgovov, fgvvvv = fermitools.lr.odc12.fancy_repulsion(
            ffoo, ffvv, goooo, govov, gvvvv, m1oo, m1vv)

    a11 = fermitools.lr.odc12.a11_sigma(
           foo, fvv, goooo, goovv, govov, gvvvv, m1oo, m1vv, t2)
    b11 = fermitools.lr.odc12.b11_sigma(
           goooo, goovv, govov, gvvvv, m1oo, m1vv, t2)
    s11 = fermitools.lr.odc12.s11_sigma(m1oo, m1vv)
    a12 = fermitools.lr.odc12.a12_sigma(gooov, govvv, fioo, fivv, t2)
    b12 = fermitools.lr.odc12.b12_sigma(gooov, govvv, fioo, fivv, t2)
    a21 = fermitools.lr.odc12.a21_sigma(gooov, govvv, fioo, fivv, t2)
    b21 = fermitools.lr.odc12.b21_sigma(gooov, govvv, fioo, fivv, t2)
    a22 = fermitools.lr.odc12.a22_sigma(
           ffoo, ffvv, goooo, govov, gvvvv, fgoooo, fgovov, fgvvvv, t2)
    b22 = fermitools.lr.odc12.b22_sigma(fgoooo, fgovov, fgvvvv, t2)

    no, nv = nocc, norb-nocc
    ns = no * nv
    nd = no * (no - 1) * nv * (nv - 1) // 4
    r1_ = fermitools.math.raveler({0: (0, 1)})
    u1_ = fermitools.math.unraveler({0: {0: no, 1: nv}})
    r2_ = fermitools.math.asym.megaraveler({0: ((0, 1), (2, 3))})
    u2_ = fermitools.math.asym.megaunraveler({0: {(0, 1): no, (2, 3): nv}})

    from toolz import functoolz

    a11_ = functoolz.compose(r1_, a11, u1_)
    a12_ = functoolz.compose(r1_, a12, u2_)
    a21_ = functoolz.compose(r2_, a21, u1_)
    a22_ = functoolz.compose(r2_, a22, u2_)
    b11_ = functoolz.compose(r1_, b11, u1_)
    b12_ = functoolz.compose(r1_, b12, u2_)
    b21_ = functoolz.compose(r2_, b21, u1_)
    b22_ = functoolz.compose(r2_, b22, u2_)
    s11_ = functoolz.compose(r1_, s11, u1_)

    a = fermitools.math.linalg.direct.bmat([[a11_, a12_], [a21_, a22_]], (ns,))
    b = fermitools.math.linalg.direct.bmat([[b11_, b12_], [b21_, b22_]], (ns,))
    s = fermitools.math.linalg.direct.block_diag(
            [s11_, fermitools.math.linalg.direct.eye], (ns,))

    e = fermitools.math.linalg.direct.bmat([[a, b], [b, a]], 2)
    m = fermitools.math.linalg.direct.block_diag(
            [s, fermitools.math.linalg.direct.negative(s)], 2)

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
    print(W_REF)
    print(U.shape)
    print(DT)
    assert_almost_equal(W[SPIN:neig], W_REF[SPIN:neig])

    nguess = neig * 2
    nvec = neig * 2
    niter = 100
    r_thresh = 1e-7
    eo = numpy.diagonal(foo)
    ev = numpy.diagonal(fvv)
    efo = numpy.diagonal(ffoo)
    efv = numpy.diagonal(ffvv)
    ad1 = r1_(fermitools.math.broadcast_sum({0: -eo, 1: +ev}))
    ad2 = r2_(fermitools.math.broadcast_sum(
        {0: -efo, 1: -efo, 2: -efv, 3: -efv}))
    sd1 = numpy.ones(ns)
    sd2 = numpy.ones(nd)
    ed = numpy.concatenate((+ad1, +ad2, +ad1, +ad2))
    md = numpy.concatenate((+sd1, +sd2, -sd1, -sd2))

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
    assert_almost_equal(w[SPIN:neig], W[SPIN:neig], decimal=10)


if __name__ == '__main__':
    main()

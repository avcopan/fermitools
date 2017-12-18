import numpy
import scipy
from toolz import functoolz
from numpy.testing import assert_almost_equal
import time

import fermitools
import solvers
import interfaces.psi4 as interface

CHARGE = +0
SPIN = 0
BASIS = '6-31+g*'
LABELS = ('O', 'H', 'H')
COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
          (0.000000000000,  1.638036840407,  1.136548822547),
          (0.000000000000, -1.638036840407,  1.136548822547))


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

    en_elec, c = solvers.oo.hf.solve(
        norb=norb, nocc=nocc, h_aso=h_aso, g_aso=g_aso, c_guess=c,
        niter=200, e_thresh=1e-14, r_thresh=1e-12, print_conv=True)
    en_nuc = fermitools.chem.nuc.energy(labels=LABELS, coords=COORDS)
    en_tot = en_elec + en_nuc
    print("\nGround state energy:")
    print('{:20.15f}'.format(en_tot))
    co, cv = numpy.split(c, (nocc,), axis=1)

    # Build the diagonal orbital and amplitude Hessian
    no = nocc
    nv = norb - nocc
    nsingles = no * nv
    hoo = fermitools.math.transform(h_aso, {0: co, 1: co})
    hvv = fermitools.math.transform(h_aso, {0: cv, 1: cv})
    goooo = fermitools.math.transform(g_aso, {0: co, 1: co, 2: co, 3: co})
    goovv = fermitools.math.transform(g_aso, {0: co, 1: co, 2: cv, 3: cv})
    govov = fermitools.math.transform(g_aso, {0: co, 1: cv, 2: co, 3: cv})
    foo = fermitools.oo.hf.fock_oo(hoo, goooo)
    fvv = fermitools.oo.hf.fock_vv(hvv, govov)
    eo = numpy.diagonal(foo)
    ev = numpy.diagonal(fvv)

    r_ = fermitools.math.raveler({0: (0, 1)})
    u_ = fermitools.math.unraveler({0: {0: no, 1: nv}})

    a_ = fermitools.lr.hf.a_sigma(foo, fvv, govov)
    b_ = fermitools.lr.hf.b_sigma(goovv)

    a_mat_ = functoolz.compose(r_, a_, u_)
    b_mat_ = functoolz.compose(r_, b_, u_)

    def e_mat_(r):
        ru, rl = numpy.split(r, 2, axis=0)
        eru = a_mat_(ru) + b_mat_(rl)
        erl = b_mat_(ru) + a_mat_(rl)
        return numpy.concatenate((eru, erl), axis=0)

    def s_mat_(r):
        ru, rl = numpy.split(r, 2, axis=0)
        sru = +ru
        srl = -rl
        return numpy.concatenate((sru, srl), axis=0)

    # Solve excitation energies
    dim = 2 * nsingles
    neig = 7
    nvec = 2 * neig
    nguess = 2 * neig
    niter = 100
    r_thresh = 1e-7

    t0 = time.time()
    e_mat = e_mat_(numpy.eye(dim))
    s_mat = s_mat_(numpy.eye(dim))
    vals, vecs = scipy.linalg.eigh(s_mat, b=e_mat)
    DT = time.time() - t0
    W = -1. / vals[:neig]
    U = vecs[:, :neig]
    print("numpy:")
    print(W)
    print(DT)

    ad = r_(fermitools.math.broadcast_sum({0: -eo, 1: +ev}))
    ed = numpy.concatenate((+ad, +ad))
    md = numpy.ones(nsingles)
    sd = numpy.concatenate((+md, -md))
    v, u, info = fermitools.math.linalg.direct.eighg(
            a=s_mat_, b=e_mat_, neig=neig, ad=sd, bd=ed, guess=U,
            r_thresh=r_thresh, nvec=nvec, niter=niter)
    w = -1. / v
    print("perfect guess:")
    print(w)
    print(info)
    assert info['niter'] == 1
    assert info['rdim'] == neig
    assert_almost_equal(w, W, decimal=10)

    t0 = time.time()
    guess = fermitools.math.linalg.direct.evec_guess(sd, nguess, bd=ed)
    v, u, info = fermitools.math.linalg.direct.eighg(
            a=s_mat_, b=e_mat_, neig=neig, ad=sd, bd=ed, guess=guess,
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

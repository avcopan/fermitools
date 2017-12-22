import numpy
import scipy
from numpy.testing import assert_almost_equal

import fermitools
import interfaces.psi4 as interface

CHARGE = +1
SPIN = 1
BASIS = 'sto-3g'
LABELS = ('O', 'H', 'H')
COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
          (0.000000000000,  1.638036840407,  1.136548822547),
          (0.000000000000, -1.638036840407,  1.136548822547))


def en_functional(no, nv, h_aso, g_aso, c):

    zoo = numpy.zeros((no, no))
    zvv = numpy.zeros((nv, nv))

    def _en(t1r, t2r):
        t1 = fermitools.math.unravel(t1r, {0: {0: no, 1: nv}})
        t2 = fermitools.math.asym.megaunravel(
                t2r, {0: {(0, 1): no, (2, 3): nv}})
        a = numpy.bmat([[zoo, -t1], [+t1.T, zvv]])
        u = scipy.linalg.expm(a)
        ct = numpy.dot(c, u)
        co, cv = numpy.split(ct, (no,), axis=1)
        hoo = fermitools.math.transform(h_aso, {0: co, 1: co})
        hvv = fermitools.math.transform(h_aso, {0: cv, 1: cv})
        goooo = fermitools.math.transform(g_aso, {0: co, 1: co, 2: co, 3: co})
        goovv = fermitools.math.transform(g_aso, {0: co, 1: co, 2: cv, 3: cv})
        govov = fermitools.math.transform(g_aso, {0: co, 1: cv, 2: co, 3: cv})
        gvvvv = fermitools.math.transform(g_aso, {0: cv, 1: cv, 2: cv, 3: cv})
        m1oo, m1vv = fermitools.oo.odc12.onebody_density(t2)
        foo = fermitools.oo.odc12.fock_xy(hoo, goooo, govov, m1oo, m1vv)
        fvv = fermitools.oo.odc12.fock_xy(hvv, govov, gvvvv, m1oo, m1vv)
        en_elec = fermitools.oo.odc12.electronic_energy(
                hoo, hvv, goooo, goovv, govov, gvvvv, m1oo, m1vv, foo, fvv, t2)
        return en_elec

    return _en


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
            niter=200, r_thresh=1e-13)
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

    a11_ = fermitools.lr.odc12.a11_sigma(
           foo, fvv, goooo, goovv, govov, gvvvv, m1oo, m1vv, t2)
    b11_ = fermitools.lr.odc12.b11_sigma(
           goooo, goovv, govov, gvvvv, m1oo, m1vv, t2)
    a12_ = fermitools.lr.odc12.a12_sigma(gooov, govvv, fioo, fivv, t2)
    b12_ = fermitools.lr.odc12.b12_sigma(gooov, govvv, fioo, fivv, t2)
    a21_ = fermitools.lr.odc12.a21_sigma(gooov, govvv, fioo, fivv, t2)
    b21_ = fermitools.lr.odc12.b21_sigma(gooov, govvv, fioo, fivv, t2)
    a22_ = fermitools.lr.odc12.a22_sigma(
           ffoo, ffvv, goooo, govov, gvvvv, fgoooo, fgovov, fgvvvv, t2)
    b22_ = fermitools.lr.odc12.b22_sigma(fgoooo, fgovov, fgvvvv, t2)

    # Print
    no, nv = nocc, norb-nocc
    n1 = no * nv
    n2 = no * (no - 1) * nv * (nv - 1) // 4
    r1_ = fermitools.math.raveler({0: (0, 1)})
    u1_ = fermitools.math.unraveler({0: {0: no, 1: nv}})
    r2_ = fermitools.math.asym.megaraveler({0: ((0, 1), (2, 3))})
    u2_ = fermitools.math.asym.megaunraveler({0: {(0, 1): no, (2, 3): nv}})
    a11 = r1_(a11_(u1_(numpy.eye(n1))))
    b11 = r1_(b11_(u1_(numpy.eye(n1))))
    a12 = r1_(a12_(u2_(numpy.eye(n2))))
    b12 = r1_(b12_(u2_(numpy.eye(n2))))
    a21 = r2_(a21_(u1_(numpy.eye(n1))))
    b21 = r2_(b21_(u1_(numpy.eye(n1))))
    a22 = r2_(a22_(u2_(numpy.eye(n2))))
    b22 = r2_(b22_(u2_(numpy.eye(n2))))
    print(a11.shape)
    print(b11.shape)
    print(a12.shape)
    print(b12.shape)
    print(a21.shape)
    print(b21.shape)
    print(a22.shape)
    print(b22.shape)

    # Zeroth derivative
    t1r = numpy.zeros(no * nv)
    t2r = fermitools.math.asym.megaravel(t2, {0: ((0, 1), (2, 3))})
    en_ = en_functional(no, nv, h_aso, g_aso, c)
    print(en_(t1r, t2r))

    # First derivatives
    from functools import partial
    en_d1 = fermitools.math.central_difference(
            f=partial(en_, t2r=t2r), x=t1r, step=0.05, nder=1, npts=11)
    print(numpy.amax(numpy.abs(en_d1)))
    en_d2 = fermitools.math.central_difference(
            f=partial(en_, t1r), x=t2r, step=0.03, nder=1, npts=17)
    print(numpy.amax(numpy.abs(en_d2)))
    assert_almost_equal(en_d1, 0., decimal=9)
    assert_almost_equal(en_d2, 0., decimal=9)

    # Second derivatives
    from functools import partial
    en_d1d1_diag = fermitools.math.central_difference(
            f=partial(en_, t2r=t2r), x=t1r, step=0.05, nder=2, npts=11)
    print(numpy.amax(numpy.abs(en_d1d1_diag - 2*numpy.diag(a11+b11))))
    assert_almost_equal(en_d1d1_diag, 2*numpy.diag(a11+b11), decimal=9)
    en_d2d2_diag = fermitools.math.central_difference(
            f=partial(en_, t1r), x=t2r, step=0.03, nder=2, npts=17)
    print(numpy.amax(numpy.abs(en_d2d2_diag - 2*numpy.diag(a22+b22))))
    assert_almost_equal(en_d2d2_diag, 2*numpy.diag(a22+b22), decimal=9)


if __name__ == '__main__':
    main()

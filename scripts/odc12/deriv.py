import numpy
import scipy
from toolz import functoolz
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
        hoo = fermitools.math.transform(h_aso, (co, co))
        hvv = fermitools.math.transform(h_aso, (cv, cv))
        goooo = fermitools.math.transform(g_aso, (co, co, co, co))
        goovv = fermitools.math.transform(g_aso, (co, co, cv, cv))
        govov = fermitools.math.transform(g_aso, (co, cv, co, cv))
        gvvvv = fermitools.math.transform(g_aso, (cv, cv, cv, cv))
        m1oo, m1vv = fermitools.oo.odc12.onebody_density(t2)
        en_elec = fermitools.oo.odc12.electronic_energy(
                hoo, hvv, goooo, goovv, govov, gvvvv, m1oo, m1vv, t2)
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
    hoo = fermitools.math.transform(h_aso, (co, co))
    hov = fermitools.math.transform(h_aso, (co, cv))
    hvv = fermitools.math.transform(h_aso, (cv, cv))
    goooo = fermitools.math.transform(g_aso, (co, co, co, co))
    gooov = fermitools.math.transform(g_aso, (co, co, co, cv))
    goovv = fermitools.math.transform(g_aso, (co, co, cv, cv))
    govov = fermitools.math.transform(g_aso, (co, cv, co, cv))
    govvv = fermitools.math.transform(g_aso, (co, cv, cv, cv))
    gvvvv = fermitools.math.transform(g_aso, (cv, cv, cv, cv))

    m1oo, m1vv = fermitools.oo.odc12.onebody_density(t2)
    foo = fermitools.oo.odc12.fock_xy(
            hxy=hoo, goxoy=goooo, gxvyv=govov, m1oo=m1oo, m1vv=m1vv)
    fov = fermitools.oo.odc12.fock_xy(
            hxy=hov, goxoy=gooov, gxvyv=govvv, m1oo=m1oo, m1vv=m1vv)
    fvv = fermitools.oo.odc12.fock_xy(
            hxy=hvv, goxoy=govov, gxvyv=gvvvv, m1oo=m1oo, m1vv=m1vv)

    a11u, b11u = fermitools.lr.odc12.onebody_hessian(
            foo, fvv, goooo, goovv, govov, gvvvv, t2)
    a12u, b12u = fermitools.lr.odc12.mixed_upper_hessian(
            fov, gooov, govvv, t2)
    a21u, b21u = fermitools.lr.odc12.mixed_lower_hessian(
            fov, gooov, govvv, t2)
    a22u, b22u = fermitools.lr.odc12.twobody_hessian(
            foo, fvv, goooo, govov, gvvvv, t2)

    # Print
    no, _, nv, _ = t2.shape
    n1 = no * nv
    n2 = no * (no - 1) * nv * (nv - 1) // 4
    r1 = fermitools.math.raveler({0: (0, 1)})
    u1 = fermitools.math.unraveler({0: {0: no, 1: nv}})
    r2 = fermitools.math.asym.megaraveler({0: ((0, 1), (2, 3))})
    u2 = fermitools.math.asym.megaunraveler({0: {(0, 1): no, (2, 3): nv}})
    a11 = functoolz.compose(r1, a11u, u1)(numpy.eye(n1))
    b11 = functoolz.compose(r1, b11u, u1)(numpy.eye(n1))
    a12 = functoolz.compose(r1, a12u, u2)(numpy.eye(n2))
    b12 = functoolz.compose(r1, b12u, u2)(numpy.eye(n2))
    a21 = functoolz.compose(r2, a21u, u1)(numpy.eye(n1))
    b21 = functoolz.compose(r2, b21u, u1)(numpy.eye(n1))
    a22 = functoolz.compose(r2, a22u, u2)(numpy.eye(n2))
    b22 = functoolz.compose(r2, b22u, u2)(numpy.eye(n2))
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
    t2r = r2(t2)
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

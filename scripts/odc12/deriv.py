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

# Ground state options
OO_NITER = 200      # number of iterations
OO_RTHRESH = 1e-12  # convergence threshold


def en_functional(no, nv, na, nb, h_ao, r_ao, c):

    zoo = numpy.zeros((no, no))
    zvv = numpy.zeros((nv, nv))

    def _en(t1r, t2r):
        t1 = fermitools.math.unravel(t1r, {0: {0: no, 1: nv}})
        t2 = fermitools.math.asym.megaunravel(
                t2r, {0: {(0, 1): no, (2, 3): nv}})
        a = numpy.bmat([[zoo, -t1], [+t1.T, zvv]])
        u = scipy.linalg.expm(a)
        au, bu = fermitools.math.spinorb.decompose_onebody(u, na=na, nb=nb)
        ac = numpy.dot(c[0], au)
        bc = numpy.dot(c[1], bu)
        aco, acv = numpy.split(ac, (na,), axis=1)
        bco, bcv = numpy.split(bc, (nb,), axis=1)
        co = (aco, bco)
        cv = (acv, bcv)
        hoo = fermitools.math.spinorb.transform_onebody(h_ao, (co, co))
        hvv = fermitools.math.spinorb.transform_onebody(h_ao, (cv, cv))
        goooo = fermitools.math.spinorb.transform_twobody(
                r_ao, (co, co, co, co))
        goovv = fermitools.math.spinorb.transform_twobody(
                r_ao, (co, co, cv, cv))
        govov = fermitools.math.spinorb.transform_twobody(
                r_ao, (co, cv, co, cv))
        gvvvv = fermitools.math.spinorb.transform_twobody(
                r_ao, (cv, cv, cv, cv))
        m1oo, m1vv = fermitools.oo.odc12.onebody_density(t2)
        en_elec = fermitools.oo.odc12.electronic_energy(
                hoo, hvv, goooo, goovv, govov, gvvvv, m1oo, m1vv, t2)
        return en_elec

    return _en


def main():
    # Spaces
    na = fermitools.chem.elec.count_alpha(LABELS, CHARGE, SPIN)
    nb = fermitools.chem.elec.count_beta(LABELS, CHARGE, SPIN)
    nbf = interface.integrals.nbf(BASIS, LABELS)
    no = na + nb
    nv = 2*nbf - no

    # Integrals
    h_ao = interface.integrals.core_hamiltonian(BASIS, LABELS, COORDS)
    r_ao = interface.integrals.repulsion(BASIS, LABELS, COORDS)

    # Mean-field guess orbitals
    c_guess = interface.hf.unrestricted_orbitals(
            BASIS, LABELS, COORDS, CHARGE, SPIN)
    t2_guess = numpy.zeros((no, no, nv, nv))

    # Solve ground state
    en_elec, c, t2, info = fermitools.oo.odc12.solve(
            na=na, nb=nb, h_ao=h_ao, r_ao=r_ao, c_guess=c_guess,
            t2_guess=t2_guess, niter=OO_NITER, r_thresh=OO_RTHRESH)
    en_nuc = fermitools.chem.nuc.energy(labels=LABELS, coords=COORDS)
    en_tot = en_elec + en_nuc
    print("\nGround state energy:")
    print('{:20.15f}'.format(en_tot))

    # LR inputs
    ac, bc = c
    aco, acv = numpy.split(ac, (na,), axis=1)
    bco, bcv = numpy.split(bc, (nb,), axis=1)
    co = (aco, bco)
    cv = (acv, bcv)
    hoo = fermitools.math.spinorb.transform_onebody(h_ao, (co, co))
    hov = fermitools.math.spinorb.transform_onebody(h_ao, (co, cv))
    hvv = fermitools.math.spinorb.transform_onebody(h_ao, (cv, cv))
    goooo = fermitools.math.spinorb.transform_twobody(r_ao, (co, co, co, co))
    gooov = fermitools.math.spinorb.transform_twobody(r_ao, (co, co, co, cv))
    goovv = fermitools.math.spinorb.transform_twobody(r_ao, (co, co, cv, cv))
    govov = fermitools.math.spinorb.transform_twobody(r_ao, (co, cv, co, cv))
    govvv = fermitools.math.spinorb.transform_twobody(r_ao, (co, cv, cv, cv))
    gvvvv = fermitools.math.spinorb.transform_twobody(r_ao, (cv, cv, cv, cv))

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
    en_ = en_functional(no, nv, na, nb, h_ao, r_ao, c)
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

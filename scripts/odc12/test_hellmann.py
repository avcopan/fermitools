import numpy
import scipy

import fermitools
import interfaces.psi4 as interface

from numpy.testing import assert_almost_equal

CHARGE = +1
SPIN = 1
BASIS = 'sto-3g'
LABELS = ('H', 'F')
COORDS = ((0., 0., 0.),
          (0., 0., 1.))


def en_f_function(h_aso, p_aso, g_aso, c_guess, t2_guess, niter=200,
                  r_thresh=1e-9, print_conv=False):

    def _en(f):
        hp_aso = h_aso - numpy.tensordot(f, p_aso, axes=(0, 0))
        en_elec, c, t2, info = fermitools.oo.odc12.solve(
                h_aso=hp_aso, g_aso=g_aso, c_guess=c_guess, t2_guess=t2_guess,
                niter=niter, r_thresh=r_thresh)

        if print_conv:
            print(info)

        return en_elec

    return _en


def test_main():
    # Spaces
    na = fermitools.chem.elec.count_alpha(LABELS, CHARGE, SPIN)
    nb = fermitools.chem.elec.count_beta(LABELS, CHARGE, SPIN)
    nocc = na + nb

    # Integrals
    nbf = interface.integrals.nbf(BASIS, LABELS)
    norb = 2 * nbf
    h_ao = interface.integrals.core_hamiltonian(BASIS, LABELS, COORDS)
    p_ao = interface.integrals.dipole(BASIS, LABELS, COORDS)
    r_ao = interface.integrals.repulsion(BASIS, LABELS, COORDS)

    h_aso = fermitools.math.spinorb.expand(h_ao, brakets=((0, 1),))
    p_aso = fermitools.math.spinorb.expand(p_ao, brakets=((1, 2),))
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

    # Differentiate
    en_f_ = en_f_function(
            h_aso=h_aso, p_aso=p_aso, g_aso=g_aso, c_guess=c, t2_guess=t2,
            niter=200, r_thresh=1e-11, print_conv=True)
    en_elec = en_f_((0., 0., 0.))
    print(en_elec)

    print("First derivative")
    en_df = fermitools.math.central_difference(
            f=en_f_, x=(0., 0., 0.), step=0.03, nder=1, npts=15)
    print(en_df)

    print("Second derivative")
    en_df2 = fermitools.math.central_difference(
            f=en_f_, x=(0., 0., 0.), step=0.03, nder=2, npts=15)
    print(en_df2)

    # LR inputs
    no, _, nv, _ = t2.shape
    co, cv = numpy.split(c, (nocc,), axis=1)
    hoo = fermitools.math.transform(h_aso, (co, co))
    hov = fermitools.math.transform(h_aso, (co, cv))
    hvv = fermitools.math.transform(h_aso, (cv, cv))
    poo = fermitools.math.transform(p_aso, (co, co))
    pov = fermitools.math.transform(p_aso, (co, cv))
    pvv = fermitools.math.transform(p_aso, (cv, cv))
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

    # Evaluate dipole moment as expectation value
    m1oo, m1vv = fermitools.oo.odc12.onebody_density(t2)
    mu = numpy.array([numpy.vdot(pxoo, m1oo) + numpy.vdot(pxvv, m1vv)
                      for pxoo, pxvv in zip(poo, pvv)])

    # Evaluate dipole polarizability by linear response
    pg = fermitools.lr.odc12.property_gradient(
            poo=poo, pov=pov, pvv=pvv, t2=t2)
    a, b = fermitools.lr.odc12.hessian(
            foo=foo, fov=fov, fvv=fvv, goooo=goooo, gooov=gooov, goovv=goovv,
            govov=govov, govvv=govvv, gvvvv=gvvvv, t2=t2)
    r = fermitools.lr.solve.static_response(a=a, b=b, pg=pg)
    alpha = numpy.dot(r.T, pg)

    print("Compare dE/df to <Psi|mu|Psi>:")
    print(en_df.round(10))
    print(mu.round(10))
    print(max(numpy.abs(en_df + mu)))

    print("Compare d2E/df2 to <<mu; mu>>:")
    print(en_df2.round(10))
    print(alpha.round(10))
    print(max(numpy.abs(en_df2 - numpy.diag(alpha))))

    assert_almost_equal(en_df, -mu, decimal=9)
    assert_almost_equal(en_df2, numpy.diag(alpha), decimal=9)


if __name__ == '__main__':
    test_main()

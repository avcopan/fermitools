import numpy

import fermitools
import interfaces.psi4 as interface

from numpy.testing import assert_almost_equal

CHARGE = +1
SPIN = 1
BASIS = 'sto-3g'
LABELS = ('H', 'F')
COORDS = ((0., 0., 0.),
          (0., 0., 1.))

# Ground state options
OO_NITER = 200      # number of iterations
OO_RTHRESH = 1e-10  # convergence threshold


def en_f_function(na, nb, h_ao, p_ao, r_ao, c_guess, t2_guess, niter=200,
                  r_thresh=1e-9):

    def _en(f):
        hp_ao = h_ao - numpy.tensordot(f, p_ao, axes=(0, 0))
        en_elec, c, t2, info = fermitools.oo.ocepa0.solve(
                na=na, nb=nb, h_ao=hp_ao, r_ao=r_ao, c_guess=c_guess,
                t2_guess=t2_guess, niter=niter, r_thresh=r_thresh,
                print_conv=False)
        print(info)
        return en_elec

    return _en


def test_main():
    # Spaces
    na = fermitools.chem.elec.count_alpha(LABELS, CHARGE, SPIN)
    nb = fermitools.chem.elec.count_beta(LABELS, CHARGE, SPIN)
    nbf = interface.integrals.nbf(BASIS, LABELS)
    no = na + nb
    nv = 2*nbf - no

    # Integrals
    h_ao = interface.integrals.core_hamiltonian(BASIS, LABELS, COORDS)
    p_ao = interface.integrals.dipole(BASIS, LABELS, COORDS)
    r_ao = interface.integrals.repulsion(BASIS, LABELS, COORDS)

    # Orbitals
    c_guess = interface.hf.unrestricted_orbitals(
            BASIS, LABELS, COORDS, CHARGE, SPIN)
    t2_guess = numpy.zeros((no, no, nv, nv))

    # Solve ground state
    en_elec, c, t2, info = fermitools.oo.ocepa0.solve(
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
    poo = fermitools.math.spinorb.transform_onebody(p_ao, (co, co))
    pov = fermitools.math.spinorb.transform_onebody(p_ao, (co, cv))
    pvv = fermitools.math.spinorb.transform_onebody(p_ao, (cv, cv))
    goooo = fermitools.math.spinorb.transform_twobody(r_ao, (co, co, co, co))
    gooov = fermitools.math.spinorb.transform_twobody(r_ao, (co, co, co, cv))
    goovv = fermitools.math.spinorb.transform_twobody(r_ao, (co, co, cv, cv))
    govov = fermitools.math.spinorb.transform_twobody(r_ao, (co, cv, co, cv))
    govvv = fermitools.math.spinorb.transform_twobody(r_ao, (co, cv, cv, cv))
    gvvvv = fermitools.math.spinorb.transform_twobody(r_ao, (cv, cv, cv, cv))
    foo = fermitools.oo.ocepa0.fock_xy(hxy=hoo, goxoy=goooo)
    fov = fermitools.oo.ocepa0.fock_xy(hxy=hov, goxoy=gooov)
    fvv = fermitools.oo.ocepa0.fock_xy(hxy=hvv, goxoy=govov)

    # Evaluate dipole moment as expectation value
    m1oo, m1vv = fermitools.oo.ocepa0.onebody_density(t2)
    mu = numpy.array([numpy.vdot(pxoo, m1oo) + numpy.vdot(pxvv, m1vv)
                      for pxoo, pxvv in zip(poo, pvv)])
    print(mu)

    # Evaluate dipole polarizability by linear response
    pg = fermitools.lr.ocepa0.property_gradient(
            poo=poo, pov=pov, pvv=pvv, t2=t2)
    a, b = fermitools.lr.ocepa0.hessian(
            foo=foo, fov=fov, fvv=fvv, goooo=goooo, gooov=gooov, goovv=goovv,
            govov=govov, govvv=govvv, gvvvv=gvvvv, t2=t2)
    r = fermitools.lr.solve.static_response(a=a, b=b, pg=pg)
    alpha = numpy.dot(r.T, pg)
    print(alpha)

    # Differentiate
    en_f_ = en_f_function(
            na=na, nb=nb, h_ao=h_ao, p_ao=p_ao, r_ao=r_ao, c_guess=c_guess,
            t2_guess=t2, niter=300, r_thresh=1e-14)
    en_elec = en_f_((0., 0., 0.))
    print(en_elec)

    print("First derivative")
    en_df = fermitools.math.central_difference(
            f=en_f_, x=(0., 0., 0.), step=0.001, nder=1, npts=5)
    print(en_df)

    print("Second derivative")
    en_df2 = fermitools.math.central_difference(
            f=en_f_, x=(0., 0., 0.), step=0.010, nder=2, npts=25)
    print(en_df2)

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

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
        en_elec, c, t2, info = fermitools.oo.ocepa0.solve(
                h_aso=hp_aso, g_aso=g_aso, c_guess=c_guess, t2_guess=t2_guess,
                niter=niter, r_thresh=r_thresh)

        if print_conv:
            print(info)

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
    en_elec, c, t2, info = fermitools.oo.ocepa0.solve(
            h_aso=h_aso, g_aso=g_aso, c_guess=c_guess, t2_guess=t2_guess,
            niter=300, r_thresh=1e-14)
    en_nuc = fermitools.chem.nuc.energy(labels=LABELS, coords=COORDS)
    en_tot = en_elec + en_nuc
    print("\nGround state energy:")
    print('{:20.15f}'.format(en_tot))

    # Differentiate
    en_f_ = en_f_function(
            h_aso=h_aso, p_aso=p_aso, g_aso=g_aso, c_guess=c, t2_guess=t2,
            niter=200, r_thresh=1e-13, print_conv=True)
    en_elec = en_f_((0., 0., 0.))
    print(en_elec)

    print("First derivative")
    en_df = fermitools.math.central_difference(
            f=en_f_, x=(0., 0., 0.), step=0.001, nder=1, npts=5)
    print(en_df)

    # LR inputs
    no, _, nv, _ = t2.shape
    co, cv = numpy.split(c, (nocc,), axis=1)
    poo = fermitools.math.transform(p_aso, {1: co, 2: co})
    pvv = fermitools.math.transform(p_aso, {1: cv, 2: cv})

    # Evaluate dipole moment as expectation value
    m1oo, m1vv = fermitools.oo.ocepa0.onebody_density(t2)
    mu = numpy.array([numpy.vdot(pxoo, m1oo) + numpy.vdot(pxvv, m1vv)
                      for pxoo, pxvv in zip(poo, pvv)])

    print("Compare dE/df to <Psi|mu|Psi>:")
    print(en_df.round(10))
    print(mu.round(10))
    print(max(numpy.abs(en_df + mu)))

    assert_almost_equal(en_df, -mu, decimal=9)


if __name__ == '__main__':
    main()

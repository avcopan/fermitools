import numpy
import scipy

import fermitools
import solvers
import interfaces.psi4 as interface
from numpy.testing import assert_almost_equal

CHARGE = +1
SPIN = 1
BASIS = 'sto-3g'
LABELS = ('O', 'H', 'H')
COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
          (0.000000000000,  1.638036840407,  1.136548822547),
          (0.000000000000, -1.638036840407,  1.136548822547))


def _main():
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
    ac, bc = interface.hf.unrestricted_orbitals(BASIS, LABELS, COORDS,
                                                CHARGE, SPIN)
    c_unsrt = scipy.linalg.block_diag(ac, bc)
    sortvec = fermitools.math.spinorb.ab2ov(dim=nbf, na=na, nb=nb)
    c_unsrt = scipy.linalg.block_diag(ac, bc)
    c = fermitools.math.spinorb.sort(c_unsrt, order=sortvec, axes=(1,))

    en_nuc = fermitools.chem.nuc.energy(labels=LABELS, coords=COORDS)
    en_elec, c = solvers.oo.hf.solve(
            norb=norb, nocc=nocc, h_aso=h_aso, g_aso=g_aso, c_guess=c,
            niter=200, e_thresh=1e-14, r_thresh=1e-12, print_conv=True)
    en_tot = en_elec + en_nuc
    print("Total energy:")
    print('{:20.15f}'.format(en_tot))

    # Evaluate dipole moment as expectation value
    p = fermitools.math.transform(p_aso, {1: c, 2: c})
    m1oo = numpy.eye(nocc)
    m1vv = numpy.zeros((norb - nocc, norb - nocc))
    m1 = scipy.linalg.block_diag(m1oo, m1vv)
    mu = numpy.array([numpy.vdot(px, m1) for px in p])

    # Evaluate dipole moment as energy derivative
    en_f = solvers.oo.hf.field_energy_solver(
            norb=norb, nocc=nocc, h_aso=h_aso, p_aso=p_aso, g_aso=g_aso,
            c_guess=c, niter=200, e_thresh=1e-13, r_thresh=1e-9,
            print_conv=True)
    en_df = fermitools.math.central_difference(en_f, (0., 0., 0.),
                                               step=0.002, npts=9)

    print("Compare dE/df to <Psi|mu|Psi>:")
    print(en_df.round(10))
    print(mu.round(10))

    assert_almost_equal(en_tot, -74.66178436045595, decimal=10)
    assert_almost_equal(en_df, -mu, decimal=11)

    # en_f = solvers.oo.hf.field_energy_solver(
    #         norb=norb, nocc=nocc, h_aso=h_aso, p_aso=p_aso, g_aso=g_aso,
    #         c_guess=c, niter=200, e_thresh=1e-13, r_thresh=1e-9,
    #         print_conv=True)
    # en_df2 = fermitools.math.central_difference(en_f, (0., 0., 0.), nder=2,
    #                                             step=0.007, npts=23)
    # print(en_df2)
    # numpy.save('en_df2', en_df2)


if __name__ == '__main__':
    _main()

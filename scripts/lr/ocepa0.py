import os
import numpy
import scipy
from numpy.testing import assert_almost_equal

import fermitools
import interfaces.psi4 as interface
import solvers

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

CHARGE = +1
SPIN = 1
BASIS = 'sto-3g'
LABELS = ('O', 'H', 'H')
COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
          (0.000000000000,  1.638036840407,  1.136548822547),
          (0.000000000000, -1.638036840407,  1.136548822547))
ALPHA_DIAG = numpy.load(os.path.join(data_path,
                                     'cation/ocepa0/alpha_diag.npy'))
EN_DF2 = numpy.load(os.path.join(data_path, 'cation/ocepa0/en_df2.npy'))
W = numpy.load(os.path.join(data_path, 'cation/ocepa0/w.npy'))


def _main():
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
    en_elec, c, t2 = solvers.oo.ocepa0.solve(
            norb=norb, nocc=nocc, h_aso=h_aso, g_aso=g_aso, c_guess=c,
            t2_guess=t2_guess, niter=200, e_thresh=1e-14, r_thresh=1e-13,
            print_conv=True)
    en_nuc = fermitools.chem.nuc.energy(labels=LABELS, coords=COORDS)
    en_tot = en_elec + en_nuc
    print("\nGround state energy:")
    print('{:20.15f}'.format(en_tot))

    # Solve response properties
    p_ao = interface.integrals.dipole(BASIS, LABELS, COORDS)
    p_aso = fermitools.math.spinorb.expand(p_ao, brakets=((1, 2),))
    alpha = solvers.lr.ocepa0.solve_static_response(
            norb=norb, nocc=nocc, h_aso=h_aso, p_aso=p_aso, g_aso=g_aso, c=c,
            t2=t2)
    print(alpha.round(8))

    assert_almost_equal(EN_DF2, numpy.diag(alpha), decimal=8)
    assert_almost_equal(ALPHA_DIAG, numpy.diag(alpha), decimal=11)

    # Solve excitation energies
    nroots = 200
    w, u = solvers.lr.ocepa0.solve_spectrum(
            norb=norb, nocc=nocc, h_aso=h_aso, g_aso=g_aso, c=c, t2=t2,
            nroots=nroots)
    print(w)
    print(u.shape)
    assert_almost_equal(W[:nroots], w, decimal=11)


if __name__ == '__main__':
    _main()

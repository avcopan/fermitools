import numpy
import scipy.linalg as spla

import fermitools

import interfaces.psi4 as interface
from . import odc12
from . import lr_ocepa0


def fancy_repulsion(ffoo, ffvv, goooo, govov, gvvvv, m1oo, m1vv):
    pass


def main():
    CHARGE = +0
    SPIN = 0
    BASIS = 'sto-3g'
    LABELS = ('O', 'H', 'H')
    COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
              (0.000000000000,  1.638036840407,  1.136548822547),
              (0.000000000000, -1.638036840407,  1.136548822547))

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
    ac, bc = interface.hf.unrestricted_orbitals(BASIS, LABELS, COORDS,
                                                CHARGE, SPIN)
    c_unsrt = spla.block_diag(ac, bc)
    sortvec = fermitools.math.spinorb.ab2ov(dim=nbf, na=na, nb=nb)
    c_unsrt = spla.block_diag(ac, bc)
    c = fermitools.math.spinorb.sort(c_unsrt, order=sortvec, axes=(1,))

    # Solve OCEPA0
    en_nuc = fermitools.chem.nuc.energy(labels=LABELS, coords=COORDS)
    t2_guess = numpy.zeros((nocc, nocc, norb-nocc, norb-nocc))
    en_elec, c, t2 = odc12.solve(norb=norb, nocc=nocc, h_aso=h_aso,
                                 g_aso=g_aso, c_guess=c,
                                 t2_guess=t2_guess, niter=200,
                                 e_thresh=1e-14, r_thresh=1e-12,
                                 print_conv=True)
    en_tot = en_elec + en_nuc
    print("Total energy:")
    print('{:20.15f}'.format(en_tot))

    # Build blocks of the electronic Hessian
    h = fermitools.math.transform(h_aso, {0: c, 1: c})
    g = fermitools.math.transform(g_aso, {0: c, 1: c, 2: c, 3: c})
    m1_ref = odc12.singles_reference_density(norb=norb, nocc=nocc)
    m1_cor = odc12.singles_correlation_density(t2)
    m1 = m1_ref + m1_cor
    k2 = odc12.doubles_cumulant(t2)
    m2 = odc12.doubles_density(m1, k2)

    a_orb = lr_ocepa0.diagonal_orbital_hessian(nocc, norb, h, g, m1, m2)
    b_orb = lr_ocepa0.offdiagonal_orbital_hessian(nocc, norb, h, g, m1, m2)

    # Get blocks of the electronic Hessian numerically
    import os
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'data')
    en_dt2 = numpy.load(os.path.join(data_path,
                                     'lr_odc12/neutral/en_dt2.npy'))
    en_dxdx = numpy.load(os.path.join(data_path,
                                      'lr_odc12/neutral/en_dxdx.npy'))

    print("Checking orbital Hessian:")
    print((en_dxdx - 2*(a_orb + b_orb)).round(8))
    print(spla.norm(en_dxdx - 2*(a_orb + b_orb)))
    print("Checking amplitude Hessian:")
    print(en_dt2.round(8))

    from numpy.testing import assert_almost_equal
    assert_almost_equal(en_dxdx, 2*(a_orb + b_orb), decimal=9)


if __name__ == '__main__':
    main()

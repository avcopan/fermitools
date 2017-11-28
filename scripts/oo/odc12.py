import os
import numpy
import scipy

import fermitools
import solvers
import interfaces.psi4 as interface
from numpy.testing import assert_almost_equal

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

CHARGE = +1
SPIN = 1
BASIS = 'sto-3g'
LABELS = ('O', 'H', 'H')
COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
          (0.000000000000,  1.638036840407,  1.136548822547),
          (0.000000000000, -1.638036840407,  1.136548822547))
EN_DF = numpy.load(os.path.join(data_path, 'odc12/en_df.npy'))


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
    ac, bc = interface.hf.unrestricted_orbitals(BASIS, LABELS, COORDS,
                                                CHARGE, SPIN)
    c_unsrt = scipy.linalg.block_diag(ac, bc)
    sortvec = fermitools.math.spinorb.ab2ov(dim=nbf, na=na, nb=nb)
    c_unsrt = scipy.linalg.block_diag(ac, bc)
    c = fermitools.math.spinorb.sort(c_unsrt, order=sortvec, axes=(1,))

    en_nuc = fermitools.chem.nuc.energy(labels=LABELS, coords=COORDS)

    t2_guess = numpy.zeros((nocc, nocc, norb-nocc, norb-nocc))
    en_elec, c, t2 = solvers.oo.odc12.solve(
            norb=norb, nocc=nocc, h_aso=h_aso, g_aso=g_aso, c_guess=c,
            t2_guess=t2_guess, niter=200, e_thresh=1e-14, r_thresh=1e-12,
            print_conv=True)
    en_tot = en_elec + en_nuc
    print("Total energy:")
    print('{:20.15f}'.format(en_tot))

    assert_almost_equal(en_tot, -74.713706346489928, decimal=10)

    # Numerically check the electronic energy gradients
    no = nocc
    nv = norb - nocc

    x = numpy.zeros(no * nv)
    t = numpy.ravel(fermitools.math.asym.ravel(t2, {0: (0, 1), 1: (2, 3)}))
    en_dx_func = solvers.oo.odc12.e_d1_f(
            norb=norb, nocc=nocc, h_aso=h_aso, g_aso=g_aso, c=c)

    print("Numerical gradient calculations ...")
    en_dx = en_dx_func(x, t)
    print("... orbital gradient finished")

    assert_almost_equal(en_dx, 0., decimal=10)

    print("Orbital gradient:")
    print(en_dx.round(8))
    print(scipy.linalg.norm(en_dx))

    # Evaluate dipole moment as expectation value
    p_ao = interface.integrals.dipole(BASIS, LABELS, COORDS)
    p_aso = fermitools.math.spinorb.expand(p_ao, brakets=((1, 2),))
    p = fermitools.math.transform(p_aso, {1: c, 2: c})
    dm1oo = numpy.eye(no)
    dm1vv = numpy.zeros((nv, nv))
    cm1oo, cm1vv = fermitools.oo.odc12.onebody_correlation_density(t2)
    dm1 = scipy.linalg.block_diag(dm1oo, dm1vv)
    cm1 = scipy.linalg.block_diag(cm1oo, cm1vv)
    m1 = dm1 + cm1
    mu = numpy.array([numpy.vdot(px, m1) for px in p])

    print("Compare dE/df to <Psi|mu|Psi>:")
    print(EN_DF.round(10))
    print(mu.round(10))

    assert_almost_equal(EN_DF, -mu, decimal=10)


if __name__ == '__main__':
    _main()

import fermitools
import fermitools.interface.pyscf as interface
from numpy.testing import assert_almost_equal


def main():
    import numpy

    CHARGE = +1
    SPIN = 1
    BASIS = 'STO-3G'
    LABELS = ('O', 'H', 'H')
    COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
              (0.000000000000,  1.638036840407,  1.136548822547),
              (0.000000000000, -1.638036840407,  1.136548822547))

    h_ao = interface.integrals.core_hamiltonian(BASIS, LABELS, COORDS)
    r_ao = interface.integrals.repulsion(BASIS, LABELS, COORDS)

    h_aso = fermitools.math.spinorb.expand(h_ao, brakets=((0, 1),))
    r_aso = fermitools.math.spinorb.expand(r_ao, brakets=((0, 2), (1, 3)))
    g_aso = r_aso - numpy.transpose(r_aso, (0, 1, 3, 2))

    na = fermitools.chem.elec.count_alpha(LABELS, CHARGE, SPIN)
    nb = fermitools.chem.elec.count_beta(LABELS, CHARGE, SPIN)
    n = na + nb

    from scipy.linalg import block_diag
    from fermitools.math.spinorb import ab2ov

    ac, bc = interface.hf.unrestricted_orbitals(BASIS, LABELS, COORDS,
                                                CHARGE, SPIN)
    c = fermitools.math.spinorb.sort(a=block_diag(ac, bc),
                                     order=ab2ov(7, na, nb),
                                     axes=(1,))

    d_aso = fermitools.hf.orb.density(n, c)

    f_aso = fermitools.hf.spohf.fock(h_aso, g_aso, d_aso)

    e = numpy.diag(fermitools.math.trans.transform(f_aso, {0: c, 1: c}))

    g = fermitools.math.trans.transform(g_aso, {0: c, 1: c, 2: c, 3: c})

    corr_energy = fermitools.corr.spinorb.mp2.correlation_energy(n=n, g=g, e=e)

    assert_almost_equal(corr_energy, -0.03588729135033, decimal=10)
    print(corr_energy)

    numpy.save('n', n)
    numpy.save('g', g)
    numpy.save('e', e)
    numpy.save('corr_energy', corr_energy)


if __name__ == '__main__':
    main()

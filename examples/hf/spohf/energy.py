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

    h = fermitools.math.spinorb.expand(h_ao, brakets=((0, 1),))
    r = fermitools.math.spinorb.expand(r_ao, brakets=((0, 2), (1, 3)))
    g = r - numpy.transpose(r, (0, 1, 3, 2))

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

    d = fermitools.hf.orb.density(n, c)

    f = fermitools.hf.spohf.fock(h, g, d)

    energy = fermitools.hf.spohf.energy(h, f, d)

    nuc_energy = fermitools.chem.nuc.energy(LABELS, COORDS)
    print(energy + nuc_energy)

    assert_almost_equal(energy, -82.664151422266826)


if __name__ == '__main__':
    main()

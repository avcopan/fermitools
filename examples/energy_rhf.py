import fermitools
import fermitools.interface.pyscf as interface
from numpy.testing import assert_almost_equal


def main():
    CHARGE = 0
    BASIS = 'STO-3G'
    LABELS = ('O', 'H', 'H')
    COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
              (0.000000000000,  1.638036840407,  1.136548822547),
              (0.000000000000, -1.638036840407,  1.136548822547))

    h = interface.integrals.core_hamiltonian(BASIS, LABELS, COORDS)
    g = interface.integrals.repulsion(BASIS, LABELS, COORDS)

    n = fermitools.chem.elec.count(LABELS, CHARGE) // 2

    c = interface.hf.restricted_orbitals(BASIS, LABELS, COORDS)

    d = fermitools.hf.orb.density(n, c)

    f = fermitools.hf.rhf.fock(h, g, d)

    energy = fermitools.hf.rhf.energy(h, f, d)

    nuc_energy = fermitools.chem.nuc.energy(LABELS, COORDS)

    assert_almost_equal(energy, -82.94444699000312, decimal=10)
    print(energy + nuc_energy)


if __name__ == '__main__':
    main()

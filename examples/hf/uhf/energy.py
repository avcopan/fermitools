import fermitools
import fermitools.interface.pyscf as interface
from numpy.testing import assert_almost_equal


def main():
    CHARGE = +1
    SPIN = 1
    BASIS = 'STO-3G'
    LABELS = ('O', 'H', 'H')
    COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
              (0.000000000000,  1.638036840407,  1.136548822547),
              (0.000000000000, -1.638036840407,  1.136548822547))

    h = interface.integrals.core_hamiltonian(BASIS, LABELS, COORDS)
    g = interface.integrals.repulsion(BASIS, LABELS, COORDS)

    na = fermitools.chem.elec.count_alpha(LABELS, CHARGE, SPIN)
    nb = fermitools.chem.elec.count_beta(LABELS, CHARGE, SPIN)

    ac, bc = interface.hf.unrestricted_orbitals(BASIS, LABELS, COORDS,
                                                CHARGE, SPIN)

    ad = fermitools.hf.orb.density(na, ac)
    bd = fermitools.hf.orb.density(nb, bc)

    af, bf = fermitools.hf.uhf.fock(h, g, ad, bd)

    energy = fermitools.hf.uhf.energy(h, af, bf, ad, bd)

    nuc_energy = fermitools.chem.nuc.energy(LABELS, COORDS)

    assert_almost_equal(energy, -82.664151422266826)
    print(energy + nuc_energy)


if __name__ == '__main__':
    main()

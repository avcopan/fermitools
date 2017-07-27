import numpy
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

    s = interface.integrals.overlap(BASIS, LABELS, COORDS)
    h = interface.integrals.core_hamiltonian(BASIS, LABELS, COORDS)
    g = interface.integrals.repulsion(BASIS, LABELS, COORDS)

    na = fermitools.chem.elec.count_alpha(LABELS, CHARGE, SPIN)
    nb = fermitools.chem.elec.count_beta(LABELS, CHARGE, SPIN)

    c_ref = interface.hf.restricted_orbitals(BASIS, LABELS, COORDS,
                                             CHARGE, SPIN)

    ad = fermitools.hf.orb.density(na, c_ref)
    bd = fermitools.hf.orb.density(nb, c_ref)

    af, bf = fermitools.hf.uhf.fock(h, g, ad, bd)
    f = fermitools.hf.rohf.effective_fock(s, af, bf, ad, bd)

    c = fermitools.hf.orb.coefficients(s, f)

    assert_almost_equal(numpy.abs(c), numpy.abs(c_ref), decimal=10)
    print(c.round(2))


if __name__ == '__main__':
    main()

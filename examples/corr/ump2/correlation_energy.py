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

    s_ao = interface.integrals.overlap(BASIS, LABELS, COORDS)
    h_ao = interface.integrals.core_hamiltonian(BASIS, LABELS, COORDS)
    g_ao = interface.integrals.repulsion(BASIS, LABELS, COORDS)

    na = fermitools.chem.elec.count_alpha(LABELS, CHARGE, SPIN)
    nb = fermitools.chem.elec.count_beta(LABELS, CHARGE, SPIN)

    ac, bc = interface.hf.unrestricted_orbitals(BASIS, LABELS, COORDS,
                                                CHARGE, SPIN)

    ad_ao = fermitools.hf.orb.density(na, ac)
    bd_ao = fermitools.hf.orb.density(nb, bc)

    af_ao, bf_ao = fermitools.hf.uhf.fock(h_ao, g_ao, ad_ao, bd_ao)

    ae = fermitools.hf.orb.energies(s_ao, af_ao)
    be = fermitools.hf.orb.energies(s_ao, bf_ao)

    aag = fermitools.math.trans.transform(g_ao, {0: ac, 1: ac, 2: ac, 3: ac})
    abg = fermitools.math.trans.transform(g_ao, {0: ac, 1: bc, 2: ac, 3: bc})
    bbg = fermitools.math.trans.transform(g_ao, {0: bc, 1: bc, 2: bc, 3: bc})

    corr_energy = fermitools.corr.ump2.correlation_energy(
            na=na, nb=nb, aag=aag, abg=abg, bbg=bbg, ae=ae, be=be)

    assert_almost_equal(corr_energy, -0.03588729135033, decimal=10)
    print(corr_energy)


if __name__ == '__main__':
    main()

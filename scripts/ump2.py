import numpy
import fermitools
import interfaces.pyscf as interface


def correlation_energy(basis, labels, coords, charge, spin):
    na = fermitools.chem.elec.count_alpha(labels, charge, spin)
    nb = fermitools.chem.elec.count_beta(labels, charge, spin)
    ao = slice(None, na)
    bo = slice(None, nb)
    av = slice(na, None)
    bv = slice(nb, None)

    ac, bc = interface.hf.unrestricted_orbitals(basis, labels, coords,
                                                charge, spin)

    aco = ac[:, ao]
    acv = ac[:, av]
    bco = bc[:, bo]
    bcv = bc[:, bv]

    h_ao = interface.integrals.core_hamiltonian(basis, labels, coords)
    g_ao = interface.integrals.repulsion(basis, labels, coords)

    ad_ao = fermitools.scf.density(ac[:, ao])
    bd_ao = fermitools.scf.density(bc[:, bo])
    af_ao, bf_ao = fermitools.scf.uhf.fock(h_ao, g_ao, ad_ao, bd_ao)

    afvv = fermitools.math.transform(af_ao, acv, acv)
    afoo = fermitools.math.transform(af_ao, aco, aco)
    bfvv = fermitools.math.transform(bf_ao, bcv, bcv)
    bfoo = fermitools.math.transform(bf_ao, bco, bco)
    aagoovv = fermitools.math.transform(g_ao, aco, aco, acv, acv)
    abgoovv = fermitools.math.transform(g_ao, aco, bco, acv, bcv)
    bbgoovv = fermitools.math.transform(g_ao, bco, bco, bcv, bcv)

    aeo = numpy.diagonal(afoo)
    aev = numpy.diagonal(afvv)
    beo = numpy.diagonal(bfoo)
    bev = numpy.diagonal(bfvv)

    aae2 = fermitools.corr.doubles_resolvent_denominator(aeo, aeo, aev, aev)
    abe2 = fermitools.corr.doubles_resolvent_denominator(aeo, beo, aev, bev)
    bbe2 = fermitools.corr.doubles_resolvent_denominator(beo, beo, bev, bev)

    aat2 = fermitools.corr.mp2.doubles_amplitudes(aagoovv, aae2)
    abt2 = fermitools.corr.mp2.doubles_amplitudes(abgoovv, abe2)
    bbt2 = fermitools.corr.mp2.doubles_amplitudes(bbgoovv, bbe2)

    return fermitools.corr.ucc.doubles_correlation_energy(aagoovv, abgoovv,
                                                          bbgoovv, aat2, abt2,
                                                          bbt2)


def main():
    from numpy.testing import assert_almost_equal

    CHARGE = +1
    SPIN = 1
    BASIS = 'STO-3G'
    LABELS = ('O', 'H', 'H')
    COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
              (0.000000000000,  1.638036840407,  1.136548822547),
              (0.000000000000, -1.638036840407,  1.136548822547))

    corr_energy = correlation_energy(BASIS, LABELS, COORDS, CHARGE, SPIN)
    print(corr_energy)

    assert_almost_equal(corr_energy, -0.03588729135033, decimal=10)


if __name__ == '__main__':
    main()

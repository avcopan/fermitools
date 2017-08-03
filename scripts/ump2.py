import numpy
import fermitools
import fermitools.interface.pyscf as interface


def t2_amplitudes(w, eo1, eo2, ev1, ev2):
    return w / fermitools.math.broadcast_sum({0: +eo1, 1: +eo2,
                                              2: -ev1, 3: -ev2})


def ump2_correlation_energy(basis, labels, coords, charge, spin):
    na = fermitools.chem.elec.count_alpha(labels, charge, spin)
    nb = fermitools.chem.elec.count_beta(labels, charge, spin)
    ao = slice(None, na)
    bo = slice(None, nb)
    av = slice(na, None)
    bv = slice(nb, None)

    ac, bc = interface.hf.unrestricted_orbitals(basis, labels, coords,
                                                charge, spin)

    h_ao = interface.integrals.core_hamiltonian(basis, labels, coords)
    g_ao = interface.integrals.repulsion(basis, labels, coords)

    ad_ao = fermitools.hf.density(ac[:, ao])
    bd_ao = fermitools.hf.density(bc[:, bo])
    af_ao, bf_ao = fermitools.hf.uhf.fock(h_ao, g_ao, ad_ao, bd_ao)

    af = fermitools.math.trans.transform(af_ao, {0: ac, 1: ac})
    bf = fermitools.math.trans.transform(bf_ao, {0: bc, 1: bc})
    aag = fermitools.math.trans.transform(g_ao, {0: ac, 1: ac, 2: ac, 3: ac})
    abg = fermitools.math.trans.transform(g_ao, {0: ac, 1: bc, 2: ac, 3: bc})
    bbg = fermitools.math.trans.transform(g_ao, {0: bc, 1: bc, 2: bc, 3: bc})

    ae = numpy.diagonal(af)
    be = numpy.diagonal(bf)

    aat2 = t2_amplitudes(aag[ao, ao, av, av], ae[ao], ae[ao], ae[av], ae[av])
    abt2 = t2_amplitudes(abg[ao, bo, av, bv], ae[ao], be[bo], ae[av], be[bv])
    bbt2 = t2_amplitudes(bbg[bo, bo, bv, bv], be[bo], be[bo], be[bv], be[bv])

    aau2 = aat2 - numpy.transpose(aat2, (0, 1, 3, 2))
    bbu2 = bbt2 - numpy.transpose(bbt2, (0, 1, 3, 2))

    return (numpy.vdot(aag[ao, ao, av, av], aau2) / 2. +
            numpy.vdot(abg[ao, bo, av, bv], abt2) +
            numpy.vdot(bbg[bo, bo, bv, bv], bbu2) / 2.)


def main():
    from numpy.testing import assert_almost_equal

    CHARGE = +1
    SPIN = 1
    BASIS = 'STO-3G'
    LABELS = ('O', 'H', 'H')
    COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
              (0.000000000000,  1.638036840407,  1.136548822547),
              (0.000000000000, -1.638036840407,  1.136548822547))

    corr_energy = ump2_correlation_energy(BASIS, LABELS, COORDS, CHARGE, SPIN)
    print(corr_energy)

    assert_almost_equal(corr_energy, -0.03588729135033, decimal=10)


if __name__ == '__main__':
    main()

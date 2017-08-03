import numpy
import fermitools
import fermitools.interface.pyscf as interface


def ump2_correlation_energy(basis, labels, coords, charge, spin):
    na = fermitools.chem.elec.count_alpha(labels, charge, spin)
    nb = fermitools.chem.elec.count_beta(labels, charge, spin)

    ac, bc = interface.hf.unrestricted_orbitals(basis, labels, coords,
                                                charge, spin)

    h_ao = interface.integrals.core_hamiltonian(basis, labels, coords)
    g_ao = interface.integrals.repulsion(basis, labels, coords)

    ad_ao = fermitools.hf.orb.density(na, ac)
    bd_ao = fermitools.hf.orb.density(nb, bc)
    af_ao, bf_ao = fermitools.hf.uhf.fock(h_ao, g_ao, ad_ao, bd_ao)

    af = fermitools.math.trans.transform(af_ao, {0: ac, 1: ac})
    bf = fermitools.math.trans.transform(bf_ao, {0: bc, 1: bc})
    aag = fermitools.math.trans.transform(g_ao, {0: ac, 1: ac, 2: ac, 3: ac})
    abg = fermitools.math.trans.transform(g_ao, {0: ac, 1: bc, 2: ac, 3: bc})
    bbg = fermitools.math.trans.transform(g_ao, {0: bc, 1: bc, 2: bc, 3: bc})

    ae = numpy.diagonal(af)
    be = numpy.diagonal(bf)

    ao = slice(None, na)
    bo = slice(None, nb)
    av = slice(na, None)
    bv = slice(nb, None)

    aat2 = aag[ao, ao, av, av] * fermitools.corr.resolvent((ae[ao], ae[ao]),
                                                           (ae[av], ae[av]))
    abt2 = abg[ao, bo, av, bv] * fermitools.corr.resolvent((ae[ao], be[bo]),
                                                           (ae[av], be[bv]))
    bbt2 = bbg[bo, bo, bv, bv] * fermitools.corr.resolvent((be[bo], be[bo]),
                                                           (be[bv], be[bv]))

    res = fermitools.corr.resolvent((ae[ao], be[bo]), (ae[av], be[bv]))
    numpy.save('ae_o', ae[ao])
    numpy.save('ae_v', ae[av])
    numpy.save('be_o', be[bo])
    numpy.save('be_v', be[bv])
    numpy.save('res', res)

    aau2 = aat2 - numpy.transpose(aat2, (0, 1, 3, 2))
    bbu2 = bbt2 - numpy.transpose(bbt2, (0, 1, 3, 2))

    return (numpy.vdot(aag[ao, ao, av, av], aau2) / 2. +
            numpy.vdot(abg[ao, bo, av, bv], abt2) +
            numpy.vdot(bbg[bo, bo, bv, bv], bbu2) / 2.)


if __name__ == '__main__':
    CHARGE = +1
    SPIN = 1
    BASIS = 'STO-3G'
    LABELS = ('O', 'H', 'H')
    COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
              (0.000000000000,  1.638036840407,  1.136548822547),
              (0.000000000000, -1.638036840407,  1.136548822547))

    corr_energy = ump2_correlation_energy(BASIS, LABELS, COORDS, CHARGE, SPIN)
    print(corr_energy)

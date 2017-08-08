import numpy
import fermitools
import interfaces.pyscf as interface


def correlation_energy(basis, labels, coords, charge):
    n = fermitools.chem.elec.count(labels, charge) // 2
    o = slice(None, n)
    v = slice(n, None)

    c = interface.hf.restricted_orbitals(basis, labels, coords, charge)

    co = c[:, o]
    cv = c[:, v]

    h_ao = interface.integrals.core_hamiltonian(basis, labels, coords)
    g_ao = interface.integrals.repulsion(basis, labels, coords)

    d_ao = fermitools.scf.density(c[:, o])
    f_ao = fermitools.scf.rhf.fock(h=h_ao, g=g_ao, d=d_ao)

    foo = fermitools.math.transform(f_ao, {0: co, 1: co})
    fvv = fermitools.math.transform(f_ao, {0: cv, 1: cv})
    goovv = fermitools.math.transform(g_ao, {0: co, 1: co, 2: cv, 3: cv})

    eo = numpy.diagonal(foo)
    ev = numpy.diagonal(fvv)

    e2 = fermitools.corr.doubles_resolvent_denominator(eo, eo, ev, ev)

    t2 = fermitools.corr.mp2.doubles_amplitudes(goovv, e2)

    return fermitools.corr.rcc.doubles_correlation_energy(goovv, t2)


def main():
    from numpy.testing import assert_almost_equal

    CHARGE = 0
    BASIS = 'STO-3G'
    LABELS = ('O', 'H', 'H')
    COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
              (0.000000000000,  1.638036840407,  1.136548822547),
              (0.000000000000, -1.638036840407,  1.136548822547))

    corr_energy = correlation_energy(BASIS, LABELS, COORDS, CHARGE)
    print(corr_energy)

    assert_almost_equal(corr_energy, -0.04914963608816, decimal=10)


if __name__ == '__main__':
    main()

import numpy
import fermitools
import fermitools.interface.pyscf as interface


def t2_amplitudes(w, eo, ev):
    return w / fermitools.math.broadcast_sum({0: +eo, 1: +eo, 2: -ev, 3: -ev})


def rmp2_correlation_energy(basis, labels, coords, charge):
    n = fermitools.chem.elec.count(labels, charge) // 2
    o = slice(None, n)
    v = slice(n, None)

    c = interface.hf.restricted_orbitals(basis, labels, coords, charge)

    h_ao = interface.integrals.core_hamiltonian(basis, labels, coords)
    g_ao = interface.integrals.repulsion(basis, labels, coords)

    d_ao = fermitools.hf.density(c[:, o])
    f_ao = fermitools.hf.rhf.fock(h=h_ao, g=g_ao, d=d_ao)

    f = fermitools.math.trans.transform(f_ao, {0: c, 1: c})
    g = fermitools.math.trans.transform(g_ao, {0: c, 1: c, 2: c, 3: c})

    e = numpy.diagonal(f)

    t2 = t2_amplitudes(g[o, o, v, v], e[o], e[v])

    u2 = 2. * t2 - numpy.transpose(t2, (0, 1, 3, 2))

    return numpy.vdot(g[o, o, v, v], u2)


def main():
    from numpy.testing import assert_almost_equal

    CHARGE = 0
    BASIS = 'STO-3G'
    LABELS = ('O', 'H', 'H')
    COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
              (0.000000000000,  1.638036840407,  1.136548822547),
              (0.000000000000, -1.638036840407,  1.136548822547))

    corr_energy = rmp2_correlation_energy(BASIS, LABELS, COORDS, CHARGE)
    print(corr_energy)

    assert_almost_equal(corr_energy, -0.04914963608816, decimal=10)


if __name__ == '__main__':
    main()

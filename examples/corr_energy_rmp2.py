import numpy
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

    s_ao = interface.integrals.overlap(BASIS, LABELS, COORDS)
    h_ao = interface.integrals.core_hamiltonian(BASIS, LABELS, COORDS)
    g_ao = interface.integrals.repulsion(BASIS, LABELS, COORDS)

    n = fermitools.chem.elec.count(LABELS, CHARGE) // 2

    c = interface.hf.restricted_orbitals(BASIS, LABELS, COORDS)

    d_ao = fermitools.hf.orb.density(n, c)
    f_ao = fermitools.hf.rhf.fock(h_ao, g_ao, d_ao)

    e = fermitools.hf.orb.energies(s_ao, f_ao)

    g = fermitools.math.trans.transform(g_ao, {0: c, 1: c, 2: c, 3: c})

    o = slice(None, n)
    v = slice(n, None)
    x = numpy.newaxis

    t = (g[o, o, v, v]
         / (e[o, x, x, x] + e[x, o, x, x] - e[x, x, v, x] - e[x, x, x, v]))
    t_eff = 2. * t - t.transpose((0, 1, 3, 2))

    corr_energy = numpy.sum(g[o, o, v, v] * t_eff)

    assert_almost_equal(corr_energy, -0.0491496361196, decimal=10)
    print(corr_energy)


if __name__ == '__main__':
    main()

import numpy
import fermitools
import fermitools.interface.pyscf as interface


def rmp2_correlation_energy(basis, labels, coords, charge):
    n = fermitools.chem.elec.count(labels, charge) // 2

    c = interface.hf.restricted_orbitals(basis, labels, coords)

    h_ao = interface.integrals.core_hamiltonian(basis, labels, coords)
    g_ao = interface.integrals.repulsion(basis, labels, coords)

    d_ao = fermitools.hf.orb.density(n, c)
    f_ao = fermitools.hf.rhf.fock(h_ao, g_ao, d_ao)

    f = fermitools.math.trans.transform(f_ao, {0: c, 1: c})
    g = fermitools.math.trans.transform(g_ao, {0: c, 1: c, 2: c, 3: c})
    e = numpy.diagonal(f)

    o = slice(None, n)
    v = slice(n, None)

    t2 = g[o, o, v, v] * fermitools.corr.resolvent((e[o], e[o]), (e[v], e[v]),)
    u2 = 2. * t2 - numpy.transpose(t2, (0, 1, 3, 2))
    return numpy.vdot(g[o, o, v, v], u2)


if __name__ == '__main__':
    CHARGE = 0
    BASIS = 'STO-3G'
    LABELS = ('O', 'H', 'H')
    COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
              (0.000000000000,  1.638036840407,  1.136548822547),
              (0.000000000000, -1.638036840407,  1.136548822547))

    corr_energy = rmp2_correlation_energy(BASIS, LABELS, COORDS, CHARGE)
    print(corr_energy)

import numpy
import fermitools
import fermitools.interface.pyscf as interface


def t2_amplitudes(w, eo, ev):
    return w / fermitools.math.broadcast_sum({0: +eo, 1: +eo, 2: -ev, 3: -ev})


def ump2_correlation_energy(basis, labels, coords, charge, spin):
    na = fermitools.chem.elec.count_alpha(labels, charge, spin)
    nb = fermitools.chem.elec.count_beta(labels, charge, spin)
    n = na + nb
    o = slice(None, n)
    v = slice(n, None)

    ac, bc = interface.hf.unrestricted_orbitals(basis, labels, coords,
                                                charge, spin)

    nbf = interface.integrals.nbf(basis, labels)
    h_ao = interface.integrals.core_hamiltonian(basis, labels, coords)
    r_ao = interface.integrals.repulsion(basis, labels, coords)

    h_aso = fermitools.math.spinorb.expand(h_ao, brakets=((0, 1),))
    r_aso = fermitools.math.spinorb.expand(r_ao, brakets=((0, 2), (1, 3)))
    g_aso = r_aso - numpy.transpose(r_aso, (0, 1, 3, 2))

    from scipy.linalg import block_diag
    from fermitools.math.spinorb import ab2ov

    c = fermitools.math.spinorb.sort(a=block_diag(ac, bc),
                                     order=ab2ov(dim=nbf, na=na, nb=nb),
                                     axes=(1,))

    d_aso = fermitools.hf.density(c[:, o])
    f_aso = fermitools.hf.spinorb.fock(h=h_aso, g=g_aso, d=d_aso)

    f = fermitools.math.trans.transform(f_aso, {0: c, 1: c})
    g = fermitools.math.trans.transform(g_aso, {0: c, 1: c, 2: c, 3: c})

    e = numpy.diagonal(f)

    t2 = t2_amplitudes(g[o, o, v, v], e[o], e[v])

    return numpy.vdot(g[o, o, v, v], t2) / 4.


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

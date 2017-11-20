import fermitools
import interfaces.psi4 as interface

import numpy
import scipy.linalg


def energy_routine(basis, labels, coords, charge, spin):
    na = fermitools.chem.elec.count_alpha(labels, charge, spin)
    nb = fermitools.chem.elec.count_beta(labels, charge, spin)
    n = na + nb
    nocc = n
    nbf = interface.integrals.nbf(basis, labels)
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

    gen = numpy.zeros((2 * nbf, 2 * nbf))
    dm1oo = numpy.eye(nocc)
    t2 = numpy.zeros_like(g_aso[o, o, v, v])

    en_nuc = fermitools.chem.nuc.energy(labels=labels, coords=coords)
    en0 = 0.
    for i in range(100):
        h = fermitools.math.transform(h_aso, {0: c, 1: c})
        g = fermitools.math.transform(g_aso, {0: c, 1: c, 2: c, 3: c})
        foo = fermitools.oo.omp2.fock_oo(h[o, o], g[o, o, o, o])
        fvv = fermitools.oo.omp2.fock_vv(h[v, v], g[o, v, o, v])

        eo = numpy.diagonal(foo)
        ev = numpy.diagonal(fvv)
        e2 = fermitools.math.broadcast_sum({0: +eo, 1: +eo,
                                            2: -ev, 3: -ev})
        r2 = fermitools.oo.omp2.twobody_amplitude_gradient(
                g[o, o, v, v], foo, fvv, t2)
        t2 += r2 / e2

        cm1oo, cm1vv = fermitools.oo.omp2.onebody_correlation_density(t2)
        m1oo = dm1oo + cm1oo
        m1vv = cm1vv
        m2oooo = fermitools.oo.omp2.twobody_moment_oooo(dm1oo, cm1oo)
        m2oovv = fermitools.oo.omp2.twobody_moment_oovv(t2)
        m2ovov = fermitools.oo.omp2.twobody_moment_ovov(dm1oo, cm1vv)

        r1 = fermitools.oo.omp2.orbital_gradient(
                h[o, v], g[o, o, o, v], g[o, v, v, v], m1oo, m1vv, m2oooo,
                m2oovv, m2ovov)
        e1 = fermitools.math.broadcast_sum({0: +eo, 1: -ev})
        t1 = r1 / e1
        gen[v, o] = numpy.transpose(t1)
        gen[o, v] = -t1
        u = scipy.linalg.expm(gen)
        c = numpy.dot(c, u)

        en_elec = fermitools.oo.omp2.electronic_energy(
                h[o, o], h[v, v], g[o, o, o, o], g[o, o, v, v], g[o, v, o, v],
                m1oo, m1vv, m2oooo, m2oovv, m2ovov)
        en = en_elec + en_nuc
        den = en - en0
        en0 = en
        print('@OMP2 {:<3d} {:20.15f} {:20.15f}'  .format(i, en, den))

    return en_elec


def main():
    CHARGE = +1
    SPIN = 1
    BASIS = 'sto-3g'
    LABELS = ('O', 'H', 'H')
    COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
              (0.000000000000,  1.638036840407,  1.136548822547),
              (0.000000000000, -1.638036840407,  1.136548822547))

    en_nuc = fermitools.chem.nuc.energy(labels=LABELS, coords=COORDS)
    en_elec = energy_routine(BASIS, LABELS, COORDS, CHARGE, SPIN)
    en_tot = en_elec + en_nuc

    print('{:20.15f}'.format(en_tot))

    from numpy.testing import assert_almost_equal
    assert_almost_equal(en_tot, -74.69827520934487, decimal=10)


if __name__ == '__main__':
    main()

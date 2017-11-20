import fermitools
import interfaces.psi4 as interface

import numpy
import scipy.linalg
from fermitools.math import einsum
from fermitools.math.asym import antisymmetrizer_product as asym


def doubles_density(dm1, cm1, k2):
    m2 = (k2
          + asym("0/1|2/3")(einsum('pr,qs->pqrs', dm1, cm1))
          + asym("2/3")(einsum('pr,qs->pqrs', dm1, dm1)))
    return m2


def doubles_cumulant(t2):
    no, _, nv, _ = t2.shape
    n = no + nv
    o = slice(None, no)
    v = slice(no, None)

    k2 = numpy.zeros((n, n, n, n))
    k2[o, o, v, v] = t2
    k2[v, v, o, o] = numpy.transpose(t2)

    return k2


def electronic_energy(h, g, m1, m2):
    return numpy.vdot(h, m1) + 1. / 4 * numpy.vdot(g, m2)


def energy_routine(basis, labels, coords, charge, spin):
    na = fermitools.chem.elec.count_alpha(labels, charge, spin)
    nb = fermitools.chem.elec.count_beta(labels, charge, spin)
    n = na + nb
    nocc = n
    nbf = interface.integrals.nbf(basis, labels)
    norb = 2 * nbf
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
    dm1vv = numpy.zeros((norb - nocc, norb - nocc))
    dm1 = scipy.linalg.block_diag(dm1oo, dm1vv)
    t2 = numpy.zeros_like(g_aso[o, o, v, v])

    en_nuc = fermitools.chem.nuc.energy(labels=labels, coords=coords)
    en0 = 0.
    for i in range(100):
        h = fermitools.math.transform(h_aso, {0: c, 1: c})
        g = fermitools.math.transform(g_aso, {0: c, 1: c, 2: c, 3: c})
        foo = fermitools.oo.fock_block(
                hxy=h[o, o], goxoy=g[o, o, o, o], m1oo=dm1[o, o])
        fvv = fermitools.oo.fock_block(
                hxy=h[v, v], goxoy=g[o, v, o, v], m1oo=dm1[o, o])

        eo = numpy.diagonal(foo)
        ev = numpy.diagonal(fvv)
        e2 = fermitools.math.broadcast_sum({0: +eo, 1: +eo,
                                            2: -ev, 3: -ev})
        r2 = fermitools.oo.omp2.twobody_amplitude_gradient(
                g[o, o, v, v], foo, fvv, t2)
        t2 += r2 / e2

        cm1oo, cm1vv = fermitools.oo.omp2.onebody_correlation_density(t2)
        cm1 = scipy.linalg.block_diag(cm1oo, cm1vv)
        m1 = dm1 + cm1
        k2 = doubles_cumulant(t2)
        m2 = doubles_density(dm1, cm1, k2)

        r1 = fermitools.oo.orbital_gradient(
                h[o, v], g[o, o, o, v], g[o, v, v, v], m1[o, o], m1[v, v],
                m2[o, o, o, o], m2[o, o, v, v], m2[o, v, o, v], m2[v, v, v, v])
        e1 = fermitools.math.broadcast_sum({0: +eo, 1: -ev})
        t1 = r1 / e1
        gen[v, o] = numpy.transpose(t1)
        gen[o, v] = -t1
        u = scipy.linalg.expm(gen)
        c = numpy.dot(c, u)

        en = electronic_energy(h, g, m1, m2) + en_nuc
        den = en - en0
        en0 = en
        print('@OMP2 {:<3d} {:20.15f} {:20.15f}'  .format(i, en, den))

    return electronic_energy(h, g, m1, m2)


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

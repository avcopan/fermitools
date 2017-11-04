import fermitools
import interfaces.psi4 as interface

import numpy
import scipy.linalg as spla
from fermitools.math.asym import antisymmetrizer_product as asym


def fock(o, h, g):
    return h + numpy.trace(g[:, o, :, o], axis1=1, axis2=3)


def doubles_numerator(goovv, foo, fvv, t2):
    foo = numpy.array(foo, copy=True)
    fvv = numpy.array(fvv, copy=True)
    numpy.fill_diagonal(foo, 0.)
    numpy.fill_diagonal(fvv, 0.)
    num2 = (goovv
            + asym("2/3")(numpy.einsum('ac,ijcb->ijab', fvv, t2))
            - asym("0/1")(numpy.einsum('ki,kjab->ijab', foo, t2)))
    return num2


def singles_correlation_density(t2):
    m1oo = - 1./2 * numpy.einsum('jkab,ikab->ij', t2, t2)
    m1vv = + 1./2 * numpy.einsum('ijac,ijbc->ab', t2, t2)
    return spla.block_diag(m1oo, m1vv)


def doubles_density(m1_ref, m1_cor, k2):
    m2 = (k2
          + asym("0/1|2/3")(numpy.einsum('pr,qs->pqrs', m1_ref, m1_cor))
          + asym("2/3")(numpy.einsum('pr,qs->pqrs', m1_ref, m1_ref)))
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


def orbital_gradient(o, v, h, g, m1, m2):
    fcap = (numpy.einsum('px,qx->pq', h, m1)
            + 1. / 2 * numpy.einsum('pxyz,qxyz->pq', g, m2))
    res1 = (numpy.transpose(fcap) - fcap)[o, v]
    return res1


def electronic_energy(h, g, m1, m2):
    return numpy.vdot(h, m1) + 1. / 4 * numpy.vdot(g, m2)


def energy_routine(basis, labels, coords, charge, spin):
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

    gen = numpy.zeros((2 * nbf, 2 * nbf))
    m1_ref = numpy.zeros_like(gen)
    m1_ref[o, o] = numpy.eye(n)
    t2 = numpy.zeros_like(g_aso[o, o, v, v])

    en_nuc = fermitools.chem.nuc.energy(labels=labels, coords=coords)
    en0 = 0.
    for i in range(100):
        h = fermitools.math.transform(h_aso, {0: c, 1: c})
        g = fermitools.math.transform(g_aso, {0: c, 1: c, 2: c, 3: c})
        f = fock(o, h, g)

        e = numpy.diagonal(f)
        e2 = fermitools.math.broadcast_sum({0: +e[o], 1: +e[o],
                                            2: -e[v], 3: -e[v]})
        t2 = doubles_numerator(g[o, o, v, v], f[o, o], f[v, v], t2) / e2

        m1_cor = singles_correlation_density(t2)
        k2 = doubles_cumulant(t2)
        m1 = m1_ref + m1_cor
        m2 = doubles_density(m1_ref, m1_cor, k2)

        r1 = orbital_gradient(o, v, h, g, m1, m2)
        e1 = fermitools.math.broadcast_sum({0: +e[o], 1: -e[v]})
        t1 = r1 / e1
        gen[v, o] = numpy.transpose(t1)
        gen[o, v] = -t1
        u = spla.expm(gen)
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

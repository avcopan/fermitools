import numpy
import scipy.linalg as spla

import fermitools
from fermitools.math import antisymmetrizer_product as asym


def fock(o, h, g):
    return h + numpy.trace(g[:, o, :, o], axis1=1, axis2=3)


def doubles_numerator(goooo, goovv, govov, gvvvv, foo, fvv, t2):
    foo = numpy.array(foo, copy=True)
    fvv = numpy.array(fvv, copy=True)
    numpy.fill_diagonal(foo, 0.)
    numpy.fill_diagonal(fvv, 0.)
    num2 = (goovv
            + asym("2/3")(numpy.einsum('ac,ijcb->ijab', fvv, t2))
            - asym("0/1")(numpy.einsum('ki,kjab->ijab', foo, t2))
            + 1. / 2 * numpy.einsum("abcd,ijcd->ijab", gvvvv, t2)
            + 1. / 2 * numpy.einsum("klij,klab->ijab", goooo, t2)
            - asym("0/1|2/3")(numpy.einsum("kaic,jkbc->ijab", govov, t2)))
    return num2


def singles_correlation_density(t2):
    m1oo = - 1./2 * numpy.einsum('jkab,ikab->ij', t2, t2)
    m1vv = + 1./2 * numpy.einsum('ijac,ijbc->ab', t2, t2)
    return spla.block_diag(m1oo, m1vv)


def doubles_cumulant(t2):
    no, _, nv, _ = t2.shape
    o = slice(None, no)
    v = slice(no, None)

    n = no + nv
    k2 = numpy.zeros((n, n, n, n))
    k2[o, o, v, v] = t2
    k2[v, v, o, o] = numpy.transpose(t2)
    k2[v, v, v, v] = 1./2 * numpy.einsum('klab,klcd->abcd', t2, t2)
    k2[o, o, o, o] = 1./2 * numpy.einsum('ijcd,klcd->ijkl', t2, t2)
    k_jabi = numpy.einsum('ikac,jkbc->jabi', t2, t2)
    k2[o, v, v, o] = +numpy.transpose(k_jabi, (0, 1, 2, 3))
    k2[o, v, o, v] = -numpy.transpose(k_jabi, (0, 1, 3, 2))
    k2[v, o, v, o] = -numpy.transpose(k_jabi, (1, 0, 2, 3))
    k2[v, o, o, v] = +numpy.transpose(k_jabi, (1, 0, 3, 2))

    return k2


def doubles_density(m1_ref, m1_cor, k2):
    m2 = (k2
          + asym("0/1|2/3")(numpy.einsum('pr,qs->pqrs', m1_ref, m1_cor))
          + asym("2/3")(numpy.einsum('pr,qs->pqrs', m1_ref, m1_ref)))
    # + asym("2/3")(numpy.einsum('pr,qs->pqrs', m1_cor, m1_cor)))
    return m2


def singles_residual(o, v, h, g, m1, m2):
    fcap = (numpy.einsum('px,qx->pq', h, m1)
            + 1. / 2 * numpy.einsum('pxyz,qxyz->pq', g, m2))
    res1 = (fcap - numpy.transpose(fcap))[o, v]
    return res1


def electronic_energy(h, g, m1, m2):
    return numpy.vdot(h, m1) + 1. / 4 * numpy.vdot(g, m2)


def energy_routine(basis, labels, coords, charge, spin):
    import interfaces.psi4 as interface

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

    from fermitools.math.spinorb import ab2ov

    c_unsrt = spla.block_diag(ac, bc)
    c = fermitools.math.spinorb.sort(c_unsrt,
                                     order=ab2ov(dim=nbf, na=na, nb=nb),
                                     axes=(1,))
    x1 = numpy.zeros((2 * nbf, 2 * nbf))
    m1_ref = numpy.zeros_like(x1)
    m1_ref[o, o] = numpy.eye(n)
    t2 = numpy.zeros_like(g_aso[o, o, v, v])

    en_elec_last = 0.
    for i in range(100):
        h = fermitools.math.transform(h_aso, {0: c, 1: c})
        g = fermitools.math.transform(g_aso, {0: c, 1: c, 2: c, 3: c})
        f = fock(o, h, g)

        e = numpy.diagonal(f)
        e2 = fermitools.math.broadcast_sum({0: +e[o], 1: +e[o],
                                            2: -e[v], 3: -e[v]})
        t2 = doubles_numerator(g[o, o, o, o], g[o, o, v, v], g[o, v, o, v],
                               g[v, v, v, v], f[o, o], f[v, v], t2) / e2
        m1_cor = singles_correlation_density(t2)
        m1 = m1_ref + m1_cor
        k2 = doubles_cumulant(t2)
        m2 = doubles_density(m1_ref, m1_cor, k2)

        r1 = singles_residual(o, v, h, g, m1, m2)
        e1 = fermitools.math.broadcast_sum({0: +e[o], 1: -e[v]})
        t1 = r1 / e1
        x1[o, v] = t1
        u = spla.expm(x1 - numpy.transpose(x1))
        c = numpy.dot(c, u)

        en_elec = electronic_energy(h, g, m1, m2)
        en_change = en_elec - en_elec_last
        en_elec_last = en_elec
        print('@OLCCD {:<3d} {:20.15f} {:20.15f}'
              .format(i, en_elec, en_change))

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
    assert_almost_equal(en_tot, -74.71451994543345, decimal=10)


if __name__ == '__main__':
    main()
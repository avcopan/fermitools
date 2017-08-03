import fermitools
import fermitools.interface.pyscf as interface
from fermitools.math.asym import antisymmetrizer

import numpy
import toolz.functoolz as ftz

A = ftz.compose(antisymmetrizer((0, 1)), antisymmetrizer((2, 3)))


def t2_resolvent_denominator(eo, ev):
    return fermitools.math.broadcast_sum({0: +eo, 1: +eo, 2: -ev, 3: -ev})


def t2_residual(o, v, g, eps, t2):
    return (- t2 * eps
            + g[o, o, v, v]
            + 1. / 2 * numpy.einsum("abcd,ijcd->ijab", g[v, v, v, v], t2)
            + 1. / 2 * numpy.einsum("klij,klab->ijab", g[o, o, o, o], t2)
            + A(numpy.einsum("akic,jkbc->ijab", g[v, o, o, v], t2)))


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

    eps = t2_resolvent_denominator(e[o], e[v])

    import functools as ft

    residual = ft.partial(t2_residual, o, v, g, eps)

    t2 = numpy.zeros((n, n, 2*nbf-n, 2*nbf-n))

    r2 = residual(t2)

    t2 = t2 + r2 / eps

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

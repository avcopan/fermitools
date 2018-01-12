import fermitools
import interfaces.pyscf as interface

import numpy


def correlation_energy(basis, labels, coords, charge, spin):
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

    co = c[:, o]
    cv = c[:, v]

    d_aso = fermitools.scf.density(co)
    f_aso = fermitools.scf.hf.fock(h=h_aso, g=g_aso, d=d_aso)

    foo = fermitools.math.transform(f_aso, co, co)
    fvv = fermitools.math.transform(f_aso, cv, cv)
    goovv = fermitools.math.transform(g_aso, co, co, cv, cv)

    eo = numpy.diagonal(foo)
    ev = numpy.diagonal(fvv)

    e2 = fermitools.corr.doubles_resolvent_denominator(eo, eo, ev, ev)

    t2 = fermitools.corr.mp2.doubles_amplitudes(goovv, e2)

    return fermitools.corr.cc.doubles_correlation_energy(goovv, t2)


def main():
    from numpy.testing import assert_almost_equal

    CHARGE = +1
    SPIN = 1
    BASIS = 'STO-3G'
    LABELS = ('O', 'H', 'H')
    COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
              (0.000000000000,  1.638036840407,  1.136548822547),
              (0.000000000000, -1.638036840407,  1.136548822547))

    corr_energy = correlation_energy(BASIS, LABELS, COORDS, CHARGE, SPIN)
    print(corr_energy)

    assert_almost_equal(corr_energy, -0.03588729135033, decimal=10)


if __name__ == '__main__':
    main()

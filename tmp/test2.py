import numpy
import scipy

import fermitools
import interfaces.psi4 as interface

CHARGE = +0
SPIN = 0
BASIS = 'sto-3g'
LABELS = ('O', 'H', 'H')
COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
          (0.000000000000,  1.638036840407,  1.136548822547),
          (0.000000000000, -1.638036840407,  1.136548822547))


def main():
    # Spaces
    na = fermitools.chem.elec.count_alpha(LABELS, CHARGE, SPIN)
    nb = fermitools.chem.elec.count_beta(LABELS, CHARGE, SPIN)
    no = na + nb

    # Integrals
    nbf = interface.integrals.nbf(BASIS, LABELS)
    h_ao = interface.integrals.core_hamiltonian(BASIS, LABELS, COORDS)
    r_ao = interface.integrals.repulsion(BASIS, LABELS, COORDS)

    h_aso = fermitools.math.spinorb.expand(h_ao, brakets=((0, 1),))
    r_aso = fermitools.math.spinorb.expand(r_ao, brakets=((0, 2), (1, 3)))
    g_aso = r_aso - numpy.transpose(r_aso, (0, 1, 3, 2))

    # Mean-field guess orbitals
    ac, bc = interface.hf.unrestricted_orbitals(
            BASIS, LABELS, COORDS, CHARGE, SPIN)
    c_unsrt = scipy.linalg.block_diag(ac, bc)
    sortvec = fermitools.math.spinorb.ab2ov(dim=nbf, na=na, nb=nb)
    print(sortvec)
    c_unsrt = scipy.linalg.block_diag(ac, bc)
    c = fermitools.math.spinorb.sort(c_unsrt, order=sortvec, axes=(1,))

    # Solve spectrum
    co, cv = numpy.split(c, (no,), axis=1)
    hoo_ = fermitools.math.transform(h_aso, (co, co))
    hov_ = fermitools.math.transform(h_aso, (co, cv))
    hvv_ = fermitools.math.transform(h_aso, (cv, cv))
    goooo_ = fermitools.math.transform(g_aso, (co, co, co, co))
    gooov_ = fermitools.math.transform(g_aso, (co, co, co, cv))
    goovv_ = fermitools.math.transform(g_aso, (co, co, cv, cv))
    govov_ = fermitools.math.transform(g_aso, (co, cv, co, cv))
    govvv_ = fermitools.math.transform(g_aso, (co, cv, cv, cv))
    gvvvv_ = fermitools.math.transform(g_aso, (cv, cv, cv, cv))
    print(hoo_.shape)
    print(hov_.shape)
    print(hvv_.shape)
    print(goooo_.shape)
    print(gooov_.shape)
    print(goovv_.shape)
    print(govov_.shape)
    print(govvv_.shape)
    print(gvvvv_.shape)

    # SDoijasdfoijasdf
    aco, acv = numpy.split(ac, (na,), axis=1)
    bco, bcv = numpy.split(bc, (nb,), axis=1)
    co = (aco, bco)
    cv = (acv, bcv)
    hoo = fermitools.math.spinorb.transform(h_ao, (co, co), brakets=((0, 1),))
    hov = fermitools.math.spinorb.transform(h_ao, (co, cv), brakets=((0, 1),))
    hvv = fermitools.math.spinorb.transform(h_ao, (cv, cv), brakets=((0, 1),))
    roooo = fermitools.math.spinorb.transform(
            r_ao, (co, co, co, co), brakets=((0, 2), (1, 3)))
    rooov = fermitools.math.spinorb.transform(
            r_ao, (co, co, co, cv), brakets=((0, 2), (1, 3)))
    roovv = fermitools.math.spinorb.transform(
            r_ao, (co, co, cv, cv), brakets=((0, 2), (1, 3)))
    rovov = fermitools.math.spinorb.transform(
            r_ao, (co, cv, co, cv), brakets=((0, 2), (1, 3)))
    rovvv = fermitools.math.spinorb.transform(
            r_ao, (co, cv, cv, cv), brakets=((0, 2), (1, 3)))
    rvvvv = fermitools.math.spinorb.transform(
            r_ao, (cv, cv, cv, cv), brakets=((0, 2), (1, 3)))
    goooo = roooo - numpy.transpose(roooo, (0, 1, 3, 2))
    print(hoo.shape)
    print(hov.shape)
    print(hvv.shape)
    print(roooo.shape)
    print(rooov.shape)
    print(roovv.shape)
    print(rovov.shape)
    print(rovvv.shape)
    print(rvvvv.shape)
    print(goooo.shape)
    from numpy.testing import assert_almost_equal
    assert_almost_equal(hoo, hoo_, decimal=14)
    assert_almost_equal(hov, hov_, decimal=14)
    assert_almost_equal(hvv, hvv_, decimal=14)
    assert_almost_equal(goooo, goooo_, decimal=14)


if __name__ == '__main__':
    main()

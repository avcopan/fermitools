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
    hoo = fermitools.math.spinorb.transform_onebody(h_ao, (co, co))
    hov = fermitools.math.spinorb.transform_onebody(h_ao, (co, cv))
    hvv = fermitools.math.spinorb.transform_onebody(h_ao, (cv, cv))
    goooo = fermitools.math.spinorb.transform_twobody(r_ao, (co, co, co, co))
    gooov = fermitools.math.spinorb.transform_twobody(r_ao, (co, co, co, cv))
    goovv = fermitools.math.spinorb.transform_twobody(r_ao, (co, co, cv, cv))
    govov = fermitools.math.spinorb.transform_twobody(r_ao, (co, cv, co, cv))
    govvv = fermitools.math.spinorb.transform_twobody(r_ao, (co, cv, cv, cv))
    gvvvv = fermitools.math.spinorb.transform_twobody(r_ao, (cv, cv, cv, cv))
    print(hoo.shape)
    print(hov.shape)
    print(hvv.shape)
    print(goooo.shape)
    print(gooov.shape)
    print(goovv.shape)
    print(govov.shape)
    print(govvv.shape)
    print(gvvvv.shape)

    from numpy.testing import assert_almost_equal
    assert_almost_equal(hoo, hoo_, decimal=14)
    assert_almost_equal(hov, hov_, decimal=14)
    assert_almost_equal(hvv, hvv_, decimal=14)
    assert_almost_equal(goooo, goooo_, decimal=14)
    assert_almost_equal(gooov, gooov_, decimal=14)
    assert_almost_equal(goovv, goovv_, decimal=14)
    assert_almost_equal(govov, govov_, decimal=14)
    assert_almost_equal(govvv, govvv_, decimal=14)
    assert_almost_equal(gvvvv, gvvvv_, decimal=14)

    h = numpy.bmat([[hoo, hov], [hov.T, hvv]])
    ah, bh = fermitools.math.spinorb.decompose_onebody(h, na, nb)
    print(ah.round(0))
    print(bh.round(0))


if __name__ == '__main__':
    main()

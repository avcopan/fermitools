import os
import numpy
import scipy
from numpy.testing import assert_almost_equal

import fermitools
import interfaces.psi4 as interface
import solvers

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

CHARGE = +1
SPIN = 1
BASIS = 'sto-3g'
LABELS = ('O', 'H', 'H')
COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
          (0.000000000000,  1.638036840407,  1.136548822547),
          (0.000000000000, -1.638036840407,  1.136548822547))
ALPHA_DIAG = numpy.load(os.path.join(data_path,
                                     'cation/ocepa0/alpha_diag.npy'))
EN_DF2 = numpy.load(os.path.join(data_path, 'cation/ocepa0/en_df2.npy'))
W = numpy.load(os.path.join(data_path, 'cation/ocepa0/w.npy'))


def _main():
    # Spaces
    na = fermitools.chem.elec.count_alpha(LABELS, CHARGE, SPIN)
    nb = fermitools.chem.elec.count_beta(LABELS, CHARGE, SPIN)
    nocc = na + nb

    # Integrals
    nbf = interface.integrals.nbf(BASIS, LABELS)
    norb = 2 * nbf
    h_ao = interface.integrals.core_hamiltonian(BASIS, LABELS, COORDS)
    r_ao = interface.integrals.repulsion(BASIS, LABELS, COORDS)

    h_aso = fermitools.math.spinorb.expand(h_ao, brakets=((0, 1),))
    r_aso = fermitools.math.spinorb.expand(r_ao, brakets=((0, 2), (1, 3)))
    g_aso = r_aso - numpy.transpose(r_aso, (0, 1, 3, 2))

    # Orbitals
    ac, bc = interface.hf.unrestricted_orbitals(
            BASIS, LABELS, COORDS, CHARGE, SPIN)
    c_unsrt = scipy.linalg.block_diag(ac, bc)
    sortvec = fermitools.math.spinorb.ab2ov(dim=nbf, na=na, nb=nb)
    c_unsrt = scipy.linalg.block_diag(ac, bc)
    c = fermitools.math.spinorb.sort(c_unsrt, order=sortvec, axes=(1,))

    # Solve OCEPA0
    t2_guess = numpy.zeros((nocc, nocc, norb-nocc, norb-nocc))
    en_elec, c, t2 = solvers.oo.ocepa0.solve(
            norb=norb, nocc=nocc, h_aso=h_aso, g_aso=g_aso, c_guess=c,
            t2_guess=t2_guess, niter=200, e_thresh=1e-14, r_thresh=1e-13,
            print_conv=True)
    en_nuc = fermitools.chem.nuc.energy(labels=LABELS, coords=COORDS)
    en_tot = en_elec + en_nuc
    print("\nGround state energy:")
    print('{:20.15f}'.format(en_tot))

    # Define LR inputs
    co, cv = numpy.split(c, (nocc,), axis=1)
    hoo = fermitools.math.transform(h_aso, {0: co, 1: co})
    hov = fermitools.math.transform(h_aso, {0: co, 1: cv})
    hvv = fermitools.math.transform(h_aso, {0: cv, 1: cv})
    goooo = fermitools.math.transform(g_aso, {0: co, 1: co, 2: co, 3: co})
    gooov = fermitools.math.transform(g_aso, {0: co, 1: co, 2: co, 3: cv})
    goovv = fermitools.math.transform(g_aso, {0: co, 1: co, 2: cv, 3: cv})
    govov = fermitools.math.transform(g_aso, {0: co, 1: cv, 2: co, 3: cv})
    govvv = fermitools.math.transform(g_aso, {0: co, 1: cv, 2: cv, 3: cv})
    gvvvv = fermitools.math.transform(g_aso, {0: cv, 1: cv, 2: cv, 3: cv})
    dm1oo = numpy.eye(nocc)
    cm1oo, cm1vv = fermitools.oo.ocepa0.onebody_correlation_density(t2)
    m1oo = dm1oo + cm1oo
    m1vv = cm1vv
    k2oooo = fermitools.oo.ocepa0.twobody_cumulant_oooo(t2)
    k2oovv = fermitools.oo.ocepa0.twobody_cumulant_oovv(t2)
    k2ovov = fermitools.oo.ocepa0.twobody_cumulant_ovov(t2)
    k2vvvv = fermitools.oo.ocepa0.twobody_cumulant_vvvv(t2)

    m2oooo = fermitools.oo.ocepa0.twobody_moment_oooo(dm1oo, cm1oo, k2oooo)
    m2oovv = fermitools.oo.ocepa0.twobody_moment_oovv(k2oovv)
    m2ovov = fermitools.oo.ocepa0.twobody_moment_ovov(dm1oo, cm1vv, k2ovov)
    m2vvvv = fermitools.oo.ocepa0.twobody_moment_vvvv(k2vvvv)

    foo = fermitools.oo.ocepa0.fock_oo(hoo, goooo)
    fov = fermitools.oo.ocepa0.fock_oo(hov, gooov)
    fvv = fermitools.oo.ocepa0.fock_vv(hvv, govov)

    a11_ = fermitools.lr.ocepa0.a11_sigma(
           hoo, hvv, goooo, goovv, govov, gvvvv, m1oo, m1vv, m2oooo, m2oovv,
           m2ovov, m2vvvv)
    b11_ = fermitools.lr.ocepa0.b11_sigma(
           goooo, goovv, govov, gvvvv, m2oooo, m2oovv, m2ovov, m2vvvv)
    a_d1d2_ = fermitools.lr.ocepa0.a_d1d2_(fov, gooov, govvv, t2)
    b_d1d2_ = fermitools.lr.ocepa0.b_d1d2_(fov, gooov, govvv, t2)
    a_d2d1_ = fermitools.lr.ocepa0.a_d2d1_(fov, gooov, govvv, t2)
    b_d2d1_ = fermitools.lr.ocepa0.b_d2d1_(fov, gooov, govvv, t2)
    a_d2d2_ = fermitools.lr.ocepa0.a_d2d2_(foo, fvv, goooo, govov, gvvvv)

    # Solve response properties
    p_ao = interface.integrals.dipole(BASIS, LABELS, COORDS)
    p_aso = fermitools.math.spinorb.expand(p_ao, brakets=((1, 2),))
    poo = fermitools.math.transform(p_aso, {1: co, 2: co})
    pov = fermitools.math.transform(p_aso, {1: co, 2: cv})
    pvv = fermitools.math.transform(p_aso, {1: cv, 2: cv})
    pg1 = fermitools.lr.ocepa0.onebody_property_gradient(pov, m1oo, m1vv)
    pg2 = fermitools.lr.ocepa0.twobody_property_gradient(poo, pvv, t2)

    alpha = solvers.lr.ocepa0.solve_static_response(
            norb=norb, nocc=nocc, a11_=a11_, b11_=b11_, a_d1d2_=a_d1d2_,
            b_d1d2_=b_d1d2_, a_d2d1_=a_d2d1_, b_d2d1_=b_d2d1_, a_d2d2_=a_d2d2_,
            pg1=pg1, pg2=pg2)
    print(alpha.round(8))

    assert_almost_equal(EN_DF2, numpy.diag(alpha), decimal=8)
    assert_almost_equal(ALPHA_DIAG, numpy.diag(alpha), decimal=11)

    # Solve excitation energies
    nroots = 200
    no, nv = nocc, norb-nocc
    s11_mat = fermitools.lr.ocepa0.s11_matrix(m1oo, m1vv)
    sinv1 = scipy.linalg.inv(s11_mat)
    sinv_d1d1 = fermitools.math.unravel(
            sinv1, {0: {0: no, 1: nv}, 1: {2: no, 3: nv}})
    x1_ = fermitools.lr.ocepa0.onebody_transformer(sinv_d1d1)
    w, u = solvers.lr.ocepa0.solve_spectrum(
            nroots=nroots, norb=norb, nocc=nocc, a11_=a11_, b11_=b11_,
            a_d1d2_=a_d1d2_, b_d1d2_=b_d1d2_, a_d2d1_=a_d2d1_, b_d2d1_=b_d2d1_,
            a_d2d2_=a_d2d2_, x1_=x1_)
    print(w)
    print(u.shape)
    assert_almost_equal(W[1:nroots], w[1:], decimal=11)


if __name__ == '__main__':
    _main()

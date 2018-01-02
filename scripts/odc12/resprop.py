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
ALPHA_DIAG = numpy.load(os.path.join(data_path, 'cation/alpha_diag.npy'))
EN_DF2 = numpy.load(os.path.join(data_path, 'cation/en_df2.npy'))
W = numpy.load(os.path.join(data_path, 'cation/w.npy'))


def main():
    # Spaces
    na = fermitools.chem.elec.count_alpha(LABELS, CHARGE, SPIN)
    nb = fermitools.chem.elec.count_beta(LABELS, CHARGE, SPIN)
    nocc = na + nb

    # Integrals
    nbf = interface.integrals.nbf(BASIS, LABELS)
    norb = 2 * nbf
    h_ao = interface.integrals.core_hamiltonian(BASIS, LABELS, COORDS)
    p_ao = interface.integrals.dipole(BASIS, LABELS, COORDS)
    r_ao = interface.integrals.repulsion(BASIS, LABELS, COORDS)

    h_aso = fermitools.math.spinorb.expand(h_ao, brakets=((0, 1),))
    p_aso = fermitools.math.spinorb.expand(p_ao, brakets=((1, 2),))
    r_aso = fermitools.math.spinorb.expand(r_ao, brakets=((0, 2), (1, 3)))
    g_aso = r_aso - numpy.transpose(r_aso, (0, 1, 3, 2))

    # Orbitals
    ac, bc = interface.hf.unrestricted_orbitals(
            BASIS, LABELS, COORDS, CHARGE, SPIN)
    c_unsrt = scipy.linalg.block_diag(ac, bc)
    sortvec = fermitools.math.spinorb.ab2ov(dim=nbf, na=na, nb=nb)
    c_unsrt = scipy.linalg.block_diag(ac, bc)
    c_guess = fermitools.math.spinorb.sort(c_unsrt, order=sortvec, axes=(1,))

    # Solve
    t2_guess = numpy.zeros((nocc, nocc, norb-nocc, norb-nocc))
    en_elec, c, t2, info = fermitools.oo.odc12.solve(
            h_aso=h_aso, g_aso=g_aso, c_guess=c_guess, t2_guess=t2_guess,
            niter=200, r_thresh=1e-14)
    en_nuc = fermitools.chem.nuc.energy(labels=LABELS, coords=COORDS)
    en_tot = en_elec + en_nuc
    print("\nGround state energy:")
    print('{:20.15f}'.format(en_tot))

    # Solve response
    no, _, nv, _ = t2.shape

    co, cv = numpy.split(c, (no,), axis=1)
    hoo = fermitools.math.transform(h_aso, {0: co, 1: co})
    hov = fermitools.math.transform(h_aso, {0: co, 1: cv})
    hvv = fermitools.math.transform(h_aso, {0: cv, 1: cv})
    poo = fermitools.math.transform(p_aso, {1: co, 2: co})
    pov = fermitools.math.transform(p_aso, {1: co, 2: cv})
    pvv = fermitools.math.transform(p_aso, {1: cv, 2: cv})
    goooo = fermitools.math.transform(g_aso, {0: co, 1: co, 2: co, 3: co})
    gooov = fermitools.math.transform(g_aso, {0: co, 1: co, 2: co, 3: cv})
    goovv = fermitools.math.transform(g_aso, {0: co, 1: co, 2: cv, 3: cv})
    govov = fermitools.math.transform(g_aso, {0: co, 1: cv, 2: co, 3: cv})
    govvv = fermitools.math.transform(g_aso, {0: co, 1: cv, 2: cv, 3: cv})
    gvvvv = fermitools.math.transform(g_aso, {0: cv, 1: cv, 2: cv, 3: cv})

    pg = fermitools.lr.odc12.property_gradient(
            poo=poo, pov=pov, pvv=pvv, t2=t2)
    a, b = fermitools.lr.odc12.hessian_sigma(
            hoo=hoo, hov=hov, hvv=hvv, goooo=goooo, gooov=gooov, goovv=goovv,
            govov=govov, govvv=govvv, gvvvv=gvvvv, t2=t2, complex=True)

    alpha_new = fermitools.lr.odc12.solve_static_response(a=a, b=b, pg=pg)
    print(alpha_new.round(10))

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
    m1oo, m1vv = fermitools.oo.odc12.onebody_density(t2)

    foo = fermitools.oo.odc12.fock_xy(
            hxy=hoo, goxoy=goooo, gxvyv=govov, m1oo=m1oo, m1vv=m1vv)
    fov = fermitools.oo.odc12.fock_xy(
            hxy=hov, goxoy=gooov, gxvyv=govvv, m1oo=m1oo, m1vv=m1vv)
    fvv = fermitools.oo.odc12.fock_xy(
            hxy=hvv, goxoy=govov, gxvyv=gvvvv, m1oo=m1oo, m1vv=m1vv)
    ffoo = fermitools.oo.odc12.fancy_property(foo, m1oo)
    ffvv = fermitools.oo.odc12.fancy_property(fvv, m1vv)

    fioo, fivv = fermitools.lr.odc12.fancy_mixed_interaction(
            fov, gooov, govvv, m1oo, m1vv)
    fgoooo, fgovov, fgvvvv = fermitools.lr.odc12.fancy_repulsion(
            ffoo, ffvv, goooo, govov, gvvvv, m1oo, m1vv)

    a11 = fermitools.lr.odc12.a11_sigma(
           foo, fvv, goooo, goovv, govov, gvvvv, m1oo, m1vv, t2)
    b11 = fermitools.lr.odc12.b11_sigma(
           goooo, goovv, govov, gvvvv, m1oo, m1vv, t2)
    a12 = fermitools.lr.odc12.a12_sigma(gooov, govvv, fioo, fivv, t2)
    b12 = fermitools.lr.odc12.b12_sigma(gooov, govvv, fioo, fivv, t2)
    a21 = fermitools.lr.odc12.a21_sigma(gooov, govvv, fioo, fivv, t2)
    b21 = fermitools.lr.odc12.b21_sigma(gooov, govvv, fioo, fivv, t2)
    a22 = fermitools.lr.odc12.a22_sigma(
           ffoo, ffvv, goooo, govov, gvvvv, fgoooo, fgovov, fgvvvv, t2)
    b22 = fermitools.lr.odc12.b22_sigma(fgoooo, fgovov, fgvvvv, t2)

    # Solve response properties
    p_ao = interface.integrals.dipole(BASIS, LABELS, COORDS)
    p_aso = fermitools.math.spinorb.expand(p_ao, brakets=((1, 2),))
    poo = fermitools.math.transform(p_aso, {1: co, 2: co})
    pov = fermitools.math.transform(p_aso, {1: co, 2: cv})
    pvv = fermitools.math.transform(p_aso, {1: cv, 2: cv})
    fpoo = fermitools.oo.odc12.fancy_property(poo, m1oo)
    fpvv = fermitools.oo.odc12.fancy_property(pvv, m1vv)
    pg1 = fermitools.lr.odc12.onebody_property_gradient(pov, m1oo, m1vv)
    pg2 = fermitools.lr.odc12.twobody_property_gradient(fpoo, -fpvv, t2)

    alpha = solvers.lr.odc12.solve_static_response(
            norb=norb, nocc=nocc, a11=a11, b11=b11, a12=a12, b12=b12, a21=a21,
            b21=b21, a22=a22, b22=b22, pg1=pg1, pg2=pg2)
    print(alpha.round(8))

    assert_almost_equal(alpha, alpha_new, decimal=10)
    assert_almost_equal(EN_DF2, numpy.diag(alpha), decimal=8)
    assert_almost_equal(ALPHA_DIAG, numpy.diag(alpha), decimal=11)

    # Save
    # numpy.save('h_aso', h_aso)
    # numpy.save('g_aso', g_aso)
    # numpy.save('c', c)
    # numpy.save('t2', t2)
    # numpy.save('hoo', hoo)
    # numpy.save('hov', hov)
    # numpy.save('hvv', hvv)
    # numpy.save('goooo', goooo)
    # numpy.save('gooov', gooov)
    # numpy.save('goovv', goovv)
    # numpy.save('govov', govov)
    # numpy.save('govvv', govvv)
    # numpy.save('gvvvv', gvvvv)
    # numpy.save('m1oo', m1oo)
    # numpy.save('m1vv', m1vv)
    # numpy.save('foo', foo)
    # numpy.save('fov', fov)
    # numpy.save('fvv', fvv)
    # numpy.save('poo', poo)
    # numpy.save('pvv', pvv)
    # numpy.save('fpoo', fpoo)
    # numpy.save('ffoo', ffoo)
    # numpy.save('ffvv', ffvv)
    # numpy.save('en_elec', en_elec)
    # numpy.save('c_guess', c_guess)
    # numpy.save('t2_guess', t2_guess)


if __name__ == '__main__':
    main()

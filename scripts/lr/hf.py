import os
import numpy
import scipy
import functools
from toolz import functoolz
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
ALPHA_DIAG = numpy.load(os.path.join(data_path, 'cation/hf/alpha_diag.npy'))
EN_DF2 = numpy.load(os.path.join(data_path, 'cation/hf/en_df2.npy'))


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

    en_elec, c = solvers.oo.hf.solve(
        norb=norb, nocc=nocc, h_aso=h_aso, g_aso=g_aso, c_guess=c,
        niter=200, e_thresh=1e-14, r_thresh=1e-12, print_conv=True)
    en_nuc = fermitools.chem.nuc.energy(labels=LABELS, coords=COORDS)
    en_tot = en_elec + en_nuc
    print("\nGround state energy:")
    print('{:20.15f}'.format(en_tot))
    co, cv = numpy.split(c, (nocc,), axis=1)

    # Build the diagonal orbital and amplitude Hessian
    no = nocc
    nv = norb - nocc
    nsingles = no * nv
    hoo = fermitools.math.transform(h_aso, {0: co, 1: co})
    hvv = fermitools.math.transform(h_aso, {0: cv, 1: cv})
    goooo = fermitools.math.transform(g_aso, {0: co, 1: co, 2: co, 3: co})
    goovv = fermitools.math.transform(g_aso, {0: co, 1: co, 2: cv, 3: cv})
    govov = fermitools.math.transform(g_aso, {0: co, 1: cv, 2: co, 3: cv})
    foo = fermitools.oo.hf.fock_oo(hoo, goooo)
    fvv = fermitools.oo.hf.fock_vv(hvv, govov)

    v1ravf = fermitools.math.raveler({0: (0, 1)})
    v1uravf = fermitools.math.unraveler({0: {0: no, 1: nv}})

    # Evaluate dipole polarizability using linear response theory
    p_ao = interface.integrals.dipole(BASIS, LABELS, COORDS)
    p_aso = fermitools.math.spinorb.expand(p_ao, brakets=((1, 2),))
    pov = fermitools.math.transform(p_aso, {1: co, 2: cv})
    pg = v1ravf(fermitools.lr.hf.pg(pov))

    a_ = fermitools.lr.hf.a_sigma(foo, fvv, govov)
    b_ = fermitools.lr.hf.b_sigma(goovv)

    e_sum_ = functoolz.compose(
            v1ravf, fermitools.func.add(a_, b_), v1uravf)
    e_dif_ = functoolz.compose(
            v1ravf, fermitools.func.sub(a_, b_), v1uravf)

    e_eff_ = functoolz.compose(e_sum_, e_dif_)

    # Response function
    n = nsingles
    e_sum_ = scipy.sparse.linalg.LinearOperator((n, n), matvec=e_sum_)
    r_solver_ = functools.partial(scipy.sparse.linalg.cg, e_sum_)
    rs, _ = zip(*map(r_solver_, -2 * numpy.moveaxis(pg, -1, 0)))
    r = numpy.moveaxis(tuple(rs), -1, 0)
    alpha = numpy.tensordot(r, pg, axes=(0, 0))
    print(alpha.round(8))

    assert_almost_equal(EN_DF2, numpy.diag(alpha), decimal=8)
    assert_almost_equal(ALPHA_DIAG, numpy.diag(alpha), decimal=11)

    # Excitation energies
    e_eff_ = scipy.sparse.linalg.LinearOperator((n, n), matvec=e_eff_)
    w2, u = scipy.sparse.linalg.eigs(e_eff_, k=n-2, which='SR')
    w = numpy.sqrt(numpy.real(sorted(w2)))
    print(w)


if __name__ == '__main__':
    _main()

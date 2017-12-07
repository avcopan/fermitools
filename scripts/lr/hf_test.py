import numpy
import scipy
import functools
from toolz import functoolz
from numpy.testing import assert_almost_equal

import fermitools
import interfaces.psi4 as interface
import solvers

CHARGE = +0
SPIN = 0
BASIS = 'sto-3g'
LABELS = ('O', 'H', 'H')
COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
          (0.000000000000,  1.638036840407,  1.136548822547),
          (0.000000000000, -1.638036840407,  1.136548822547))
W = [
        0.2851637170, 0.2851637170, 0.2851637170, 0.2997434467, 0.2997434467,
        0.2997434467, 0.3526266606, 0.3526266606, 0.3526266606, 0.3547782530,
        0.3651313107, 0.3651313107, 0.3651313107, 0.4153174946, 0.5001011401,
        0.5106610509, 0.5106610509, 0.5106610509, 0.5460719086, 0.5460719086,
        0.5460719086, 0.5513718846, 0.6502707118, 0.8734253708, 1.1038187957,
        1.1038187957, 1.1038187957, 1.1957870714, 1.1957870714, 1.1957870714,
        1.2832053178, 1.3237421886, 19.9585040647, 19.9585040647,
        19.9585040647, 20.0109471551, 20.0113074586, 20.0113074586,
        20.0113074586, 20.0504919449]


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
    eo = numpy.diagonal(foo)
    ev = numpy.diagonal(fvv)

    r_ = fermitools.math.raveler({0: (0, 1)})
    u_ = fermitools.math.unraveler({0: {0: no, 1: nv}})

    # Evaluate dipole polarizability using linear response theory
    p_ao = interface.integrals.dipole(BASIS, LABELS, COORDS)
    p_aso = fermitools.math.spinorb.expand(p_ao, brakets=((1, 2),))
    pov = fermitools.math.transform(p_aso, {1: co, 2: cv})
    pg = r_(fermitools.lr.hf.pg(pov))

    a_ = fermitools.lr.hf.a_sigma(foo, fvv, govov)
    b_ = fermitools.lr.hf.b_sigma(goovv)
    pc_ = fermitools.lr.hf.pc_sigma(eo, ev)

    e_sum_ = functoolz.compose(
            r_, fermitools.func.add(a_, b_), u_)
    e_dif_ = functoolz.compose(
            r_, fermitools.func.sub(a_, b_), u_)

    e_eff_ = functoolz.compose(e_sum_, e_dif_)

    # Response function
    n = nsingles
    e_sum_ = scipy.sparse.linalg.LinearOperator((n, n), matvec=e_sum_)
    r_solver_ = functools.partial(scipy.sparse.linalg.cg, e_sum_)
    rs, _ = zip(*map(r_solver_, -2 * numpy.moveaxis(pg, -1, 0)))
    r = numpy.moveaxis(tuple(rs), -1, 0)
    alpha = numpy.tensordot(r, pg, axes=(0, 0))
    print(alpha.round(8))

    # Excitation energies
    nroots = 5
    e_eff_ = scipy.sparse.linalg.LinearOperator((n, n), matvec=e_eff_)
    w2, u = scipy.sparse.linalg.eigs(e_eff_, k=nroots, which='SR')
    w = numpy.sqrt(numpy.real(sorted(w2)))
    print(w)
    assert_almost_equal(w, W[:nroots])

    # w_td, u_td = scipy.linalg.eigh(r_(a_(u_(numpy.eye(n)))))
    # x = u_td[:, :nroots]
    # y = numpy.zeros_like(x)
    x = numpy.eye(n)[:, :nroots]
    y = numpy.zeros_like(x)

    a = functoolz.compose(r_, a_, u_)
    b = functoolz.compose(r_, b_, u_)

    def pc(w):
        return functoolz.compose(r_, pc_(w), u_)

    w_new = solvers.lr.hf.solve_spectrum(
            nroots=nroots, a=a, b=b, pc=pc, x_guess=x, y_guess=y)
    print(w_new)
    assert_almost_equal(w_new, W[:nroots])


if __name__ == '__main__':
    _main()

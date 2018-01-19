import numpy
import scipy

import fermitools
import interfaces.psi4 as interface

import os
from numpy.testing import assert_almost_equal

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

W_REF = numpy.load(os.path.join(data_path, 'neutral/w.npy'))

CHARGE = +0
SPIN = 0
BASIS = 'sto-3g'
LABELS = ('O', 'H', 'H')
COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
          (0.000000000000,  1.638036840407,  1.136548822547),
          (0.000000000000, -1.638036840407,  1.136548822547))

# Ground state options
OO_NITER = 200      # number of iterations
OO_RTHRESH = 1e-10  # convergence threshold

# Excited state options
LR_NROOT = 7        # number of roots
LR_NGUESS = 2       # number of guess vectors per root
LR_NVEC = 20        # number of subspace vectors per root
LR_NITER = 200      # number of iterations
LR_RTHRESH = 1e-5   # convergence threshold


def test_main():
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

    # Mean-field guess orbitals
    ac, bc = interface.hf.unrestricted_orbitals(
            BASIS, LABELS, COORDS, CHARGE, SPIN)
    c_unsrt = scipy.linalg.block_diag(ac, bc)
    sortvec = fermitools.math.spinorb.ab2ov(dim=nbf, na=na, nb=nb)
    c_unsrt = scipy.linalg.block_diag(ac, bc)
    c_guess = fermitools.math.spinorb.sort(c_unsrt, order=sortvec, axes=(1,))
    t2_guess = numpy.zeros((nocc, nocc, norb-nocc, norb-nocc))

    # Solve ground state
    en_elec, c, t2, info = fermitools.oo.odc12.solve(
            h_aso=h_aso, g_aso=g_aso, c_guess=c_guess, t2_guess=t2_guess,
            niter=OO_NITER, r_thresh=OO_RTHRESH)
    en_nuc = fermitools.chem.nuc.energy(labels=LABELS, coords=COORDS)
    en_tot = en_elec + en_nuc
    print("\nGround state energy:")
    print('{:20.15f}'.format(en_tot))
    assert_almost_equal(en_tot, -75.013041342026796, decimal=10)

    # Solve spectrum
    no, _, nv, _ = t2.shape
    co, cv = numpy.split(c, (no,), axis=1)
    hoo = fermitools.math.transform(h_aso, co, co)
    hov = fermitools.math.transform(h_aso, co, cv)
    hvv = fermitools.math.transform(h_aso, cv, cv)
    goooo = fermitools.math.transform(g_aso, co, co, co, co)
    gooov = fermitools.math.transform(g_aso, co, co, co, cv)
    goovv = fermitools.math.transform(g_aso, co, co, cv, cv)
    govov = fermitools.math.transform(g_aso, co, cv, co, cv)
    govvv = fermitools.math.transform(g_aso, co, cv, cv, cv)
    gvvvv = fermitools.math.transform(g_aso, cv, cv, cv, cv)

    m1oo, m1vv = fermitools.oo.odc12.onebody_density(t2)
    foo = fermitools.oo.odc12.fock_xy(
            hxy=hoo, goxoy=goooo, gxvyv=govov, m1oo=m1oo, m1vv=m1vv)
    fov = fermitools.oo.odc12.fock_xy(
            hxy=hov, goxoy=gooov, gxvyv=govvv, m1oo=m1oo, m1vv=m1vv)
    fvv = fermitools.oo.odc12.fock_xy(
            hxy=hvv, goxoy=govov, gxvyv=gvvvv, m1oo=m1oo, m1vv=m1vv)

    sd = fermitools.lr.odc12.metric_zeroth_order_diagonal(no, nv)
    ad = fermitools.lr.odc12.hessian_zeroth_order_diagonal(
            foo=foo, fvv=fvv, t2=t2)

    s, d = fermitools.lr.odc12.metric(t2=t2)
    a, b = fermitools.lr.odc12.hessian(
            foo=foo, fov=fov, fvv=fvv, goooo=goooo, gooov=gooov, goovv=goovv,
            govov=govov, govvv=govvv, gvvvv=gvvvv, t2=t2)

    w, u, info = fermitools.lr.solve.spectrum(
            a=a, b=b, s=s, d=d, ad=ad, sd=sd, nroot=LR_NROOT, nguess=LR_NGUESS,
            nvec=LR_NVEC, niter=LR_NITER, r_thresh=LR_RTHRESH)
    print(w)
    assert_almost_equal(w[:LR_NROOT], W_REF[:LR_NROOT], decimal=10)


if __name__ == '__main__':
    test_main()

#    from pycallgraph import PyCallGraph
#    from pycallgraph import Config
#    from pycallgraph import GlobbingFilter
#    from pycallgraph.output import GraphvizOutput
#    config = Config()
#    config.trace_filter = GlobbingFilter(include=['fermitools.*'])
#    graphviz = GraphvizOutput(output_file='filtered.png')
#    with PyCallGraph(output=graphviz, config=config):
#        test_main()

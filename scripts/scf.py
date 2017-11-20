import numpy
import scipy.linalg

import warnings

import fermitools
from fermitools.math import einsum
from fermitools.math.asym import antisymmetrizer_product as asym

import interfaces.psi4 as interface


def doubles_density(m1):
    m2 = asym("2/3")(einsum('pr,qs->pqrs', m1, m1))
    return m2


def energy_functional(norb, nocc, h_aso, g_aso, c):
    o = slice(None, nocc)
    v = slice(nocc, None)
    no = nocc
    nv = norb - nocc

    m1oo = numpy.eye(no)
    m1vv = numpy.zeros((nv, nv))
    m1 = scipy.linalg.block_diag(m1oo, m1vv)
    m2 = doubles_density(m1)

    def _energy(t1_flat):
        t1 = numpy.reshape(t1_flat, (no, nv))
        gen = numpy.zeros((norb, norb))
        gen[v, o] = numpy.transpose(t1)
        gen[o, v] = -t1
        u = scipy.linalg.expm(gen)
        ct = numpy.dot(c, u)

        h = fermitools.math.transform(h_aso, {0: ct, 1: ct})
        g = fermitools.math.transform(g_aso, {0: ct, 1: ct, 2: ct, 3: ct})

        en_elec = fermitools.oo.electronic_energy(
                h[o, o], h[v, v], g[o, o, o, o], g[o, o, v, v], g[o, v, o, v],
                g[v, v, v, v], m1[o, o], m1[v, v], m2[o, o, o, o],
                m2[o, o, v, v], m2[o, v, o, v], m2[v, v, v, v])
        return en_elec

    return _energy


def orbital_gradient_functional(norb, nocc, h_aso, g_aso, c, step=0.05,
                                npts=9):
    en_func = energy_functional(norb, nocc, h_aso, g_aso, c)

    def _orbital_gradient(t1_flat):
        en_dx = fermitools.math.central_difference(en_func, t1_flat,
                                                   step=step, nder=1,
                                                   npts=npts)
        return en_dx

    return _orbital_gradient


def orbital_hessian_functional(norb, nocc, h_aso, g_aso, c, step=0.05,
                               npts=9):
    en_dx_func = orbital_gradient_functional(norb, nocc, h_aso, g_aso, c,
                                             step=step, npts=npts)

    def _orbital_hessian(t1_flat):
        en_dxdx = fermitools.math.central_difference(en_dx_func, t1_flat,
                                                     step=step, nder=1,
                                                     npts=npts)
        return en_dxdx

    return _orbital_hessian


def solve(norb, nocc, h_aso, g_aso, c_guess, niter=50, e_thresh=1e-10,
          r_thresh=1e-9, print_conv=False):
    o = slice(None, nocc)
    v = slice(nocc, None)

    c = c_guess
    gen = numpy.zeros((norb, norb))

    m1oo = numpy.eye(nocc)
    m1vv = numpy.zeros((norb - nocc, norb - nocc))
    m1 = scipy.linalg.block_diag(m1oo, m1vv)
    m2 = doubles_density(m1)

    en_elec_last = 0.
    for iteration in range(niter):
        h = fermitools.math.transform(h_aso, {0: c, 1: c})
        g = fermitools.math.transform(g_aso, {0: c, 1: c, 2: c, 3: c})
        foo = fermitools.oo.fock_block(
                hxy=h[o, o], goxoy=g[o, o, o, o], m1oo=m1[o, o])
        fvv = fermitools.oo.fock_block(
                hxy=h[v, v], goxoy=g[o, v, o, v], m1oo=m1[o, o])

        eo = numpy.diagonal(foo)
        ev = numpy.diagonal(fvv)

        r1 = fermitools.oo.orbital_gradient(
                h[o, v], g[o, o, o, v], g[o, v, v, v], m1[o, o], m1[v, v],
                m2[o, o, o, o], m2[o, o, v, v], m2[o, v, o, v], m2[v, v, v, v])
        e1 = fermitools.math.broadcast_sum({0: +eo, 1: -ev})
        t1 = r1 / e1
        gen[v, o] = numpy.transpose(t1)
        gen[o, v] = -t1
        u = scipy.linalg.expm(gen)
        c = numpy.dot(c, u)

        en_elec = fermitools.oo.electronic_energy(
                h[o, o], h[v, v], g[o, o, o, o], g[o, o, v, v], g[o, v, o, v],
                g[v, v, v, v], m1[o, o], m1[v, v], m2[o, o, o, o],
                m2[o, o, v, v], m2[o, v, o, v], m2[v, v, v, v])
        en_change = en_elec - en_elec_last
        en_elec_last = en_elec

        r_norm = scipy.linalg.norm(r1)

        converged = (numpy.fabs(en_change) < e_thresh and r_norm < r_thresh)

        if converged:
            break

    if not converged:
        warnings.warn("Did not converge! (dE: {:7.1e}, r_norm: {:7.1e})"
                      .format(en_change, r_norm))

    if print_conv:
        print("E={:20.15f} ({:-3d} iterations, dE: {:7.1e}, r_norm: {:7.1e})"
              .format(en_elec, iteration, en_change, r_norm))

    return en_elec, c


def perturbed_energy_function(norb, nocc, h_aso, p_aso, g_aso, c_guess,
                              niter=50, e_thresh=1e-10, r_thresh=1e-9,
                              print_conv=False):

    def _energy(f=(0., 0., 0.)):
        hp_aso = h_aso - numpy.tensordot(f, p_aso, axes=(0, 0))
        en_elec, c = solve(norb=norb, nocc=nocc, h_aso=hp_aso,
                           g_aso=g_aso, c_guess=c_guess, niter=niter,
                           e_thresh=e_thresh, r_thresh=r_thresh,
                           print_conv=print_conv)
        return en_elec

    return _energy


def main():
    CHARGE = +1
    SPIN = 1
    BASIS = 'sto-3g'
    LABELS = ('O', 'H', 'H')
    COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
              (0.000000000000,  1.638036840407,  1.136548822547),
              (0.000000000000, -1.638036840407,  1.136548822547))

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
    ac, bc = interface.hf.unrestricted_orbitals(BASIS, LABELS, COORDS,
                                                CHARGE, SPIN)
    c_unsrt = scipy.linalg.block_diag(ac, bc)
    sortvec = fermitools.math.spinorb.ab2ov(dim=nbf, na=na, nb=nb)
    c_unsrt = scipy.linalg.block_diag(ac, bc)
    c = fermitools.math.spinorb.sort(c_unsrt, order=sortvec, axes=(1,))

    en_nuc = fermitools.chem.nuc.energy(labels=LABELS, coords=COORDS)
    en_elec, c = solve(norb=norb, nocc=nocc, h_aso=h_aso, g_aso=g_aso,
                       c_guess=c, niter=200, e_thresh=1e-14,
                       r_thresh=1e-12, print_conv=True)
    en_tot = en_elec + en_nuc
    print("Total energy:")
    print('{:20.15f}'.format(en_tot))

    # Evaluate dipole moment as expectation value
    p = fermitools.math.transform(p_aso, {1: c, 2: c})
    m1oo = numpy.eye(nocc)
    m1vv = numpy.zeros((norb - nocc, norb - nocc))
    m1 = scipy.linalg.block_diag(m1oo, m1vv)
    mu = numpy.array([numpy.vdot(px, m1) for px in p])

    # Evaluate dipole moment as energy derivative
    en_f = perturbed_energy_function(norb=norb, nocc=nocc, h_aso=h_aso,
                                     p_aso=p_aso, g_aso=g_aso, c_guess=c,
                                     niter=200, e_thresh=1e-13, r_thresh=1e-9,
                                     print_conv=True)
    en_df = fermitools.math.central_difference(en_f, (0., 0., 0.),
                                               step=0.002, npts=9)

    print("Compare dE/df to <Psi|mu|Psi>:")
    print(en_df.round(10))
    print(mu.round(10))

    from numpy.testing import assert_almost_equal
    assert_almost_equal(en_tot, -74.66178436045595, decimal=10)
    assert_almost_equal(en_df, -mu, decimal=11)


if __name__ == '__main__':
    main()

import numpy
import scipy.linalg as spla
import functools

import warnings

import fermitools
from fermitools.math import einsum
from fermitools.math.asym import antisymmetrizer_product as asym

import interfaces.psi4 as interface


def fock(h, g, m1):
    return h + numpy.tensordot(g, m1, axes=((1, 3), (0, 1)))


def doubles_numerator(goooo, goovv, govov, gvvvv, foo, fvv, t2):
    foo = numpy.array(foo, copy=True)
    fvv = numpy.array(fvv, copy=True)
    numpy.fill_diagonal(foo, 0.)
    numpy.fill_diagonal(fvv, 0.)
    num2 = (goovv
            + asym("2/3")(einsum('ac,ijcb->ijab', fvv, t2))
            - asym("0/1")(einsum('ki,kjab->ijab', foo, t2))
            + 1. / 2 * einsum("abcd,ijcd->ijab", gvvvv, t2)
            + 1. / 2 * einsum("klij,klab->ijab", goooo, t2)
            - asym("0/1|2/3")(einsum("kaic,jkbc->ijab", govov, t2)))
    return num2


def singles_reference_density(norb, nocc):
    m1_ref = numpy.zeros((norb, norb))
    m1_ref[:nocc, :nocc] = numpy.eye(nocc)
    return m1_ref


def singles_correlation_density(t2):
    m1oo = - 1./2 * einsum('jkab,ikab->ij', t2, t2)
    m1vv = + 1./2 * einsum('ijac,ijbc->ab', t2, t2)
    return spla.block_diag(m1oo, m1vv)


def doubles_cumulant(t2):
    no, _, nv, _ = t2.shape
    o = slice(None, no)
    v = slice(no, None)

    n = no + nv
    k2 = numpy.zeros((n, n, n, n))
    k2[o, o, v, v] = t2
    k2[v, v, o, o] = numpy.transpose(t2)
    k2[v, v, v, v] = 1./2 * einsum('klab,klcd->abcd', t2, t2)
    k2[o, o, o, o] = 1./2 * einsum('ijcd,klcd->ijkl', t2, t2)
    k_jabi = einsum('ikac,jkbc->jabi', t2, t2)
    k2[o, v, v, o] = +numpy.transpose(k_jabi, (0, 1, 2, 3))
    k2[o, v, o, v] = -numpy.transpose(k_jabi, (0, 1, 3, 2))
    k2[v, o, v, o] = -numpy.transpose(k_jabi, (1, 0, 2, 3))
    k2[v, o, o, v] = +numpy.transpose(k_jabi, (1, 0, 3, 2))

    return k2


def doubles_density(m1_ref, m1_cor, k2):
    m2 = (k2
          + asym("0/1|2/3")(einsum('pr,qs->pqrs', m1_ref, m1_cor))
          + asym("2/3")(einsum('pr,qs->pqrs', m1_ref, m1_ref)))
    return m2


def first_order_orbital_variation_matrix(h, g, m1, m2):
    fc = (einsum('px,qx->pq', h, m1)
          + 1. / 2 * einsum('pxyz,qxyz->pq', g, m2))
    return fc


def orbital_gradient(o, v, h, g, m1, m2):
    fc = first_order_orbital_variation_matrix(h, g, m1, m2)
    res1 = (numpy.transpose(fc) - fc)[o, v]
    return res1


def electronic_energy(h, g, m1, m2):
    return numpy.vdot(h, m1) + 1. / 4 * numpy.vdot(g, m2)


def solve(norb, nocc, h_aso, g_aso, c_guess, t2_guess, niter=50,
          e_thresh=1e-10, r_thresh=1e-9, print_conv=False):
    o = slice(None, nocc)
    v = slice(nocc, None)

    gen = numpy.zeros((norb, norb))
    m1_ref = singles_reference_density(norb=norb, nocc=nocc)

    c = c_guess
    t2_last = t2_guess
    en_elec_last = 0.
    for iteration in range(niter):
        h = fermitools.math.transform(h_aso, {0: c, 1: c})
        g = fermitools.math.transform(g_aso, {0: c, 1: c, 2: c, 3: c})
        f = fock(h, g, m1_ref)

        e = numpy.diagonal(f)
        e2 = fermitools.math.broadcast_sum({0: +e[o], 1: +e[o],
                                            2: -e[v], 3: -e[v]})
        t2 = doubles_numerator(g[o, o, o, o], g[o, o, v, v], g[o, v, o, v],
                               g[v, v, v, v], f[o, o], f[v, v], t2_last) / e2
        r2 = (t2 - t2_last) * e2
        t2_last = t2
        m1_cor = singles_correlation_density(t2)
        m1 = m1_ref + m1_cor
        k2 = doubles_cumulant(t2)
        m2 = doubles_density(m1_ref, m1_cor, k2)

        r1 = orbital_gradient(o, v, h, g, m1, m2)
        e1 = fermitools.math.broadcast_sum({0: +e[o], 1: -e[v]})
        t1 = r1 / e1
        gen[v, o] = numpy.transpose(t1)
        gen[o, v] = -t1
        u = spla.expm(gen)
        c = numpy.dot(c, u)

        en_elec = electronic_energy(h, g, m1, m2)
        en_change = en_elec - en_elec_last
        en_elec_last = en_elec

        r_norm = spla.norm([spla.norm(r1), spla.norm(r2)])

        converged = (numpy.fabs(en_change) < e_thresh and r_norm < r_thresh)

        if converged:
            break

    if not converged:
        warnings.warn("Did not converge! (dE: {:7.1e}, r_norm: {:7.1e})"
                      .format(en_change, r_norm))

    if print_conv:
        print("E={:20.15f} ({:-3d} iterations, dE: {:7.1e}, r_norm: {:7.1e})"
              .format(en_elec, iteration, en_change, r_norm))

    return en_elec, c, t2


def energy_functional(norb, nocc, h_aso, g_aso, c):
    o = slice(None, nocc)
    v = slice(nocc, None)
    no = nocc
    nv = norb - nocc
    noo = no * (no - 1) // 2
    nvv = nv * (nv - 1) // 2

    gen = numpy.zeros((norb, norb))
    m1_ref = singles_reference_density(norb=norb, nocc=nocc)

    def _electronic_energy(t1_flat, t2_flat):
        t1 = numpy.reshape(t1_flat, (no, nv))
        t2_mat = numpy.reshape(t2_flat, (noo, nvv))
        t2 = fermitools.math.asym.unravel_compound_index(t2_mat, {0: (0, 1),
                                                                  1: (2, 3)})
        gen[v, o] = numpy.transpose(t1)
        gen[o, v] = -t1
        u = spla.expm(gen)
        ct = numpy.dot(c, u)

        h = fermitools.math.transform(h_aso, {0: ct, 1: ct})
        g = fermitools.math.transform(g_aso, {0: ct, 1: ct, 2: ct, 3: ct})

        m1_cor = singles_correlation_density(t2)
        m1 = m1_ref + m1_cor
        k2 = doubles_cumulant(t2)
        m2 = doubles_density(m1_ref, m1_cor, k2)

        return electronic_energy(h, g, m1, m2)

    return _electronic_energy


def orbital_gradient_functional(norb, nocc, h_aso, g_aso, c, step=0.01,
                                npts=9):
    en_func = energy_functional(norb, nocc, h_aso, g_aso, c)

    def _orbital_gradient(t1_flat, t2_flat):
        en_dx = fermitools.math.central_difference(
                    functools.partial(en_func, t2_flat=t2_flat), t1_flat,
                    step=step, nder=1, npts=npts)
        return en_dx

    return _orbital_gradient


def amplitude_gradient_functional(norb, nocc, h_aso, g_aso, c, step=0.01,
                                  npts=9):
    en_func = energy_functional(norb, nocc, h_aso, g_aso, c)

    def _amplitude_gradient(t1_flat, t2_flat):
        en_dt = fermitools.math.central_difference(
                    functools.partial(en_func, t1_flat), t2_flat, step=step,
                    nder=1, npts=npts)
        return en_dt

    return _amplitude_gradient


def orbital_hessian_functional(norb, nocc, h_aso, g_aso, c, step=0.01, npts=9):
    en_dx_func = orbital_gradient_functional(norb, nocc, h_aso, g_aso, c,
                                             step=step, npts=npts)

    def _orbital_hessian(t1_flat, t2_flat):
        en_dxdx = fermitools.math.central_difference(
                    functools.partial(en_dx_func, t2_flat=t2_flat), t1_flat,
                    step=step, nder=1, npts=npts)
        return en_dxdx

    return _orbital_hessian


def mixed_hessian_functional(norb, nocc, h_aso, g_aso, c, step=0.01, npts=9):
    en_dt_func = amplitude_gradient_functional(norb, nocc, h_aso, g_aso, c,
                                               step=step, npts=npts)

    def _mixed_hessian(t1_flat, t2_flat):
        en_dxdt = fermitools.math.central_difference(
                    functools.partial(en_dt_func, t2_flat=t2_flat), t1_flat,
                    step=step, nder=1, npts=npts)
        return en_dxdt

    return _mixed_hessian


def mixed_hessian_transp_functional(norb, nocc, h_aso, g_aso, c, step=0.01,
                                    npts=9):
    en_dx_func = orbital_gradient_functional(norb, nocc, h_aso, g_aso, c,
                                             step=step, npts=npts)

    def _mixed_hessian(t1_flat, t2_flat):
        en_dtdx = fermitools.math.central_difference(
                    functools.partial(en_dx_func, t1_flat), t2_flat, step=step,
                    nder=1, npts=npts)
        return en_dtdx

    return _mixed_hessian


def amplitude_hessian_functional(norb, nocc, h_aso, g_aso, c, step=0.01,
                                 npts=9):
    en_dt_func = amplitude_gradient_functional(norb, nocc, h_aso, g_aso, c,
                                               step=step, npts=npts)

    def _amplitude_hessian(t1_flat, t2_flat):
        en_dtdt = fermitools.math.central_difference(
                    functools.partial(en_dt_func, t1_flat), t2_flat, step=step,
                    nder=1, npts=npts)
        return en_dtdt

    return _amplitude_hessian


def perturbed_energy_function(norb, nocc, h_aso, p_aso, g_aso, c_guess,
                              t2_guess, niter=50, e_thresh=1e-10,
                              r_thresh=1e-9, print_conv=False):

    def _electronic_energy(f=(0., 0., 0.)):
        hp_aso = h_aso - numpy.tensordot(f, p_aso, axes=(0, 0))
        en_elec, c, t2 = solve(norb=norb, nocc=nocc, h_aso=hp_aso,
                               g_aso=g_aso, c_guess=c_guess,
                               t2_guess=t2_guess, niter=niter,
                               e_thresh=e_thresh, r_thresh=r_thresh,
                               print_conv=print_conv)
        return en_elec

    return _electronic_energy


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
    r_ao = interface.integrals.repulsion(BASIS, LABELS, COORDS)

    h_aso = fermitools.math.spinorb.expand(h_ao, brakets=((0, 1),))
    r_aso = fermitools.math.spinorb.expand(r_ao, brakets=((0, 2), (1, 3)))
    g_aso = r_aso - numpy.transpose(r_aso, (0, 1, 3, 2))

    # Orbitals
    ac, bc = interface.hf.unrestricted_orbitals(BASIS, LABELS, COORDS,
                                                CHARGE, SPIN)
    c_unsrt = spla.block_diag(ac, bc)
    sortvec = fermitools.math.spinorb.ab2ov(dim=nbf, na=na, nb=nb)
    c_unsrt = spla.block_diag(ac, bc)
    c = fermitools.math.spinorb.sort(c_unsrt, order=sortvec, axes=(1,))

    en_nuc = fermitools.chem.nuc.energy(labels=LABELS, coords=COORDS)

    t2_guess = numpy.zeros((nocc, nocc, norb-nocc, norb-nocc))
    en_elec, c, t2 = solve(norb=norb, nocc=nocc, h_aso=h_aso,
                           g_aso=g_aso, c_guess=c, t2_guess=t2_guess,
                           niter=200, e_thresh=1e-14, r_thresh=1e-12,
                           print_conv=True)
    en_tot = en_elec + en_nuc
    print("Total energy:")
    print('{:20.15f}'.format(en_tot))

    from numpy.testing import assert_almost_equal
    assert_almost_equal(en_tot, -74.71451994543345, decimal=10)

    # Numerically check the electronic energy gradients
    no = nocc
    nv = norb - nocc

    x = numpy.zeros(no * nv)
    t = numpy.ravel(fermitools.math.asym.compound_index(t2, {0: (0, 1),
                                                             1: (2, 3)}))
    en_dx_func = orbital_gradient_functional(norb=norb, nocc=nocc,
                                             h_aso=h_aso, g_aso=g_aso,
                                             c=c, npts=11)
    en_dt_func = amplitude_gradient_functional(norb=norb, nocc=nocc,
                                               h_aso=h_aso, g_aso=g_aso,
                                               c=c, npts=11)

    print("Numerical gradient calculations ...")
    en_dx = en_dx_func(x, t)
    print("... orbital gradient finished")
    en_dt = en_dt_func(x, t)
    print("... amplitude gradient finished")

    print("Orbital gradient:")
    print(en_dx.round(8))
    print(spla.norm(en_dx))

    print("Amplitude gradient:")
    print(en_dt.round(8))
    print(spla.norm(en_dt))

    # Evaluate dipole moment as expectation value
    p_ao = interface.integrals.dipole(BASIS, LABELS, COORDS)
    p_aso = fermitools.math.spinorb.expand(p_ao, brakets=((1, 2),))
    p = fermitools.math.transform(p_aso, {1: c, 2: c})
    m1_ref = singles_reference_density(norb=norb, nocc=nocc)
    m1_cor = singles_correlation_density(t2)
    m1 = m1_ref + m1_cor
    mu = numpy.array([numpy.vdot(px, m1) for px in p])

    # Evaluate dipole moment as energy derivative
    en_f = perturbed_energy_function(norb=norb, nocc=nocc, h_aso=h_aso,
                                     p_aso=p_aso, g_aso=g_aso, c_guess=c,
                                     t2_guess=t2, niter=200, e_thresh=1e-13,
                                     r_thresh=1e-9, print_conv=True)
    en_df = fermitools.math.central_difference(en_f, (0., 0., 0.),
                                               step=0.002, npts=9)
    print("Compare dE/df to <Psi|mu|Psi>:")
    print(en_df.round(10))
    print(mu.round(10))

    assert_almost_equal(en_df, -mu, decimal=10)


if __name__ == '__main__':
    main()

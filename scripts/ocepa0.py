import numpy
import scipy.linalg
import functools

import warnings

import fermitools

import interfaces.psi4 as interface


def solve(norb, nocc, h_aso, g_aso, c_guess, t2_guess, niter=50,
          e_thresh=1e-10, r_thresh=1e-9, print_conv=False):
    o = slice(None, nocc)
    v = slice(nocc, None)

    gen = numpy.zeros((norb, norb))
    dm1oo = numpy.eye(nocc)

    c = c_guess
    t2 = t2_guess
    en_elec_last = 0.
    for iteration in range(niter):
        h = fermitools.math.transform(h_aso, {0: c, 1: c})
        g = fermitools.math.transform(g_aso, {0: c, 1: c, 2: c, 3: c})
        foo = fermitools.oo.ocepa0.fock_oo(h[o, o], g[o, o, o, o])
        fvv = fermitools.oo.ocepa0.fock_vv(h[v, v], g[o, v, o, v])
        eo = numpy.diagonal(foo)
        ev = numpy.diagonal(fvv)
        e2 = fermitools.math.broadcast_sum({0: +eo, 1: +eo,
                                            2: -ev, 3: -ev})
        r2 = fermitools.oo.ocepa0.twobody_amplitude_gradient(
                g[o, o, o, o], g[o, o, v, v], g[o, v, o, v],
                g[v, v, v, v], foo, fvv, t2)
        t2 += r2 / e2
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

        r1 = fermitools.oo.ocepa0.orbital_gradient(
                h[o, v], g[o, o, o, v], g[o, v, v, v], m1oo, m1vv,
                m2oooo, m2oovv, m2ovov, m2vvvv)
        e1 = fermitools.math.broadcast_sum({0: +eo, 1: -ev})
        t1 = r1 / e1
        gen[v, o] = numpy.transpose(t1)
        gen[o, v] = -t1
        u = scipy.linalg.expm(gen)
        c = numpy.dot(c, u)

        en_elec = fermitools.oo.ocepa0.electronic_energy(
                h[o, o], h[v, v], g[o, o, o, o], g[o, o, v, v], g[o, v, o, v],
                g[v, v, v, v], m1oo, m1vv, m2oooo, m2oovv, m2ovov,
                m2vvvv)
        en_change = en_elec - en_elec_last
        en_elec_last = en_elec

        r_norm = scipy.linalg.norm(
                [scipy.linalg.norm(r1), scipy.linalg.norm(r2)])

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
    dm1oo = numpy.eye(no)

    def _electronic_energy(t1_flat, t2_flat):
        t1 = numpy.reshape(t1_flat, (no, nv))
        t2_mat = numpy.reshape(t2_flat, (noo, nvv))
        t2 = fermitools.math.asym.unravel(t2_mat, {0: (0, 1), 1: (2, 3)})
        gen[v, o] = numpy.transpose(t1)
        gen[o, v] = -t1
        u = scipy.linalg.expm(gen)
        ct = numpy.dot(c, u)

        h = fermitools.math.transform(h_aso, {0: ct, 1: ct})
        g = fermitools.math.transform(g_aso, {0: ct, 1: ct, 2: ct, 3: ct})
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

        en_elec = fermitools.oo.ocepa0.electronic_energy(
                h[o, o], h[v, v], g[o, o, o, o], g[o, o, v, v], g[o, v, o, v],
                g[v, v, v, v], m1oo, m1vv, m2oooo, m2oovv, m2ovov,
                m2vvvv)
        return en_elec

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
    c_unsrt = scipy.linalg.block_diag(ac, bc)
    sortvec = fermitools.math.spinorb.ab2ov(dim=nbf, na=na, nb=nb)
    c_unsrt = scipy.linalg.block_diag(ac, bc)
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
    t = numpy.ravel(fermitools.math.asym.ravel(t2, {0: (0, 1), 1: (2, 3)}))
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
    print(scipy.linalg.norm(en_dx))

    print("Amplitude gradient:")
    print(en_dt.round(8))
    print(scipy.linalg.norm(en_dt))

    # Evaluate dipole moment as expectation value
    p_ao = interface.integrals.dipole(BASIS, LABELS, COORDS)
    p_aso = fermitools.math.spinorb.expand(p_ao, brakets=((1, 2),))
    p = fermitools.math.transform(p_aso, {1: c, 2: c})
    dm1oo = numpy.eye(no)
    dm1vv = numpy.zeros((nv, nv))
    cm1oo, cm1vv = fermitools.oo.ocepa0.onebody_correlation_density(t2)
    dm1 = scipy.linalg.block_diag(dm1oo, dm1vv)
    cm1 = scipy.linalg.block_diag(cm1oo, cm1vv)
    m1 = dm1 + cm1
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

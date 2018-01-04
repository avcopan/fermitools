import numpy
import scipy
import warnings
import functools

import fermitools


def solve(norb, nocc, h_aso, g_aso, c_guess, t2_guess, niter=50,
          e_thresh=1e-10, r_thresh=1e-9, print_conv=False):
    no, _, nv, _ = t2_guess.shape

    zoo = numpy.zeros((no, no))
    zvv = numpy.zeros((nv, nv))
    dm1oo = numpy.eye(no)

    c = c_guess
    t2 = t2_guess
    en_elec_last = 0.
    for iteration in range(niter):
        co, cv = numpy.split(c, (no,), axis=1)
        hoo = fermitools.math.transform(h_aso, {0: co, 1: co})
        hov = fermitools.math.transform(h_aso, {0: co, 1: cv})
        hvv = fermitools.math.transform(h_aso, {0: cv, 1: cv})
        goooo = fermitools.math.transform(g_aso, {0: co, 1: co, 2: co, 3: co})
        gooov = fermitools.math.transform(g_aso, {0: co, 1: co, 2: co, 3: cv})
        goovv = fermitools.math.transform(g_aso, {0: co, 1: co, 2: cv, 3: cv})
        govov = fermitools.math.transform(g_aso, {0: co, 1: cv, 2: co, 3: cv})
        govvv = fermitools.math.transform(g_aso, {0: co, 1: cv, 2: cv, 3: cv})
        gvvvv = fermitools.math.transform(g_aso, {0: cv, 1: cv, 2: cv, 3: cv})
        foo = fermitools.oo.ocepa0.fock_xy(hoo, goooo)
        fvv = fermitools.oo.ocepa0.fock_xy(hvv, govov)
        eo = numpy.diagonal(foo)
        ev = numpy.diagonal(fvv)
        e2 = fermitools.math.broadcast_sum({0: +eo, 1: +eo,
                                            2: -ev, 3: -ev})
        r2 = fermitools.oo.ocepa0.twobody_amplitude_gradient(
                goooo, goovv, govov, gvvvv, foo, fvv, t2)
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
                hov, gooov, govvv, m1oo, m1vv, m2oooo, m2oovv, m2ovov, m2vvvv)
        e1 = fermitools.math.broadcast_sum({0: +eo, 1: -ev})
        t1 = r1 / e1
        a = numpy.bmat([[zoo, -t1], [+t1.T, zvv]])
        u = scipy.linalg.expm(a)
        c = numpy.dot(c, u)

        en_elec = fermitools.oo.ocepa0.electronic_energy(
                hoo, hvv, goooo, goovv, govov, gvvvv, m1oo, m1vv, m2oooo,
                m2oovv, m2ovov, m2vvvv)
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


def field_energy_solver(norb, nocc, h_aso, p_aso, g_aso, c_guess, t2_guess,
                        niter=50, e_thresh=1e-10, r_thresh=1e-9,
                        print_conv=False):

    def _electronic_energy(f=(0., 0., 0.)):
        hp_aso = h_aso - numpy.tensordot(f, p_aso, axes=(0, 0))
        en_elec, c, t2 = solve(norb=norb, nocc=nocc, h_aso=hp_aso,
                               g_aso=g_aso, c_guess=c_guess,
                               t2_guess=t2_guess, niter=niter,
                               e_thresh=e_thresh, r_thresh=r_thresh,
                               print_conv=print_conv)
        return en_elec

    return _electronic_energy


def e_f(norb, nocc, h_aso, g_aso, c):
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
        t2 = fermitools.math.asym.unravel(t2_mat, {0: ((0, 1), no),
                                                   1: ((2, 3), nv)})
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


def e_d1_f(norb, nocc, h_aso, g_aso, c, step=0.01, npts=9):
    en_func = e_f(norb, nocc, h_aso, g_aso, c)

    def _orbital_gradient(t1_flat, t2_flat):
        en_dx = fermitools.math.central_difference(
                    functools.partial(en_func, t2_flat=t2_flat), t1_flat,
                    step=step, nder=1, npts=npts)
        return en_dx

    return _orbital_gradient


def e_d2_f(norb, nocc, h_aso, g_aso, c, step=0.01, npts=9):
    en_func = e_f(norb, nocc, h_aso, g_aso, c)

    def _amplitude_gradient(t1_flat, t2_flat):
        en_dt = fermitools.math.central_difference(
                    functools.partial(en_func, t1_flat), t2_flat, step=step,
                    nder=1, npts=npts)
        return en_dt

    return _amplitude_gradient


def e_2d2_f(norb, nocc, h_aso, g_aso, c, step=0.01, npts=9):
    en_func = e_f(norb, nocc, h_aso, g_aso, c)

    def _amplitude_hessian_diag(t1_flat, t2_flat):
        en_dt2 = fermitools.math.central_difference(
                    functools.partial(en_func, t1_flat), t2_flat, step=step,
                    nder=2, npts=npts)
        return en_dt2

    return _amplitude_hessian_diag


def e_d1d1_f(norb, nocc, h_aso, g_aso, c, step=0.01, npts=9):
    en_dx_func = e_d1_f(norb, nocc, h_aso, g_aso, c, step=step, npts=npts)

    def _orbital_hessian(t1_flat, t2_flat):
        en_dxdx = fermitools.math.central_difference(
                    functools.partial(en_dx_func, t2_flat=t2_flat), t1_flat,
                    step=step, nder=1, npts=npts)
        return en_dxdx

    return _orbital_hessian


def e_d1d2_f(norb, nocc, h_aso, g_aso, c, step=0.01, npts=9):
    en_dt_func = e_d2_f(norb, nocc, h_aso, g_aso, c, step=step, npts=npts)

    def _mixed_hessian(t1_flat, t2_flat):
        en_dxdt = fermitools.math.central_difference(
                    functools.partial(en_dt_func, t2_flat=t2_flat), t1_flat,
                    step=step, nder=1, npts=npts)
        return en_dxdt

    return _mixed_hessian


def e_d2d1_f(norb, nocc, h_aso, g_aso, c, step=0.01, npts=9):
    en_dx_func = e_d1_f(norb, nocc, h_aso, g_aso, c, step=step, npts=npts)

    def _mixed_hessian(t1_flat, t2_flat):
        en_dtdx = fermitools.math.central_difference(
                    functools.partial(en_dx_func, t1_flat), t2_flat, step=step,
                    nder=1, npts=npts)
        return en_dtdx

    return _mixed_hessian


def e_d2d2_f(norb, nocc, h_aso, g_aso, c, step=0.01, npts=9):
    en_dt_func = e_d2_f(norb, nocc, h_aso, g_aso, c, step=step, npts=npts)

    def _amplitude_hessian(t1_flat, t2_flat):
        en_dtdt = fermitools.math.central_difference(
                    functools.partial(en_dt_func, t1_flat), t2_flat, step=step,
                    nder=1, npts=npts)
        return en_dtdt

    return _amplitude_hessian

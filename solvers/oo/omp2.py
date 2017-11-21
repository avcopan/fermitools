import numpy
import scipy
import warnings

import fermitools


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
        foo = fermitools.oo.omp2.fock_oo(h[o, o], g[o, o, o, o])
        fvv = fermitools.oo.omp2.fock_vv(h[v, v], g[o, v, o, v])

        eo = numpy.diagonal(foo)
        ev = numpy.diagonal(fvv)
        e2 = fermitools.math.broadcast_sum({0: +eo, 1: +eo,
                                            2: -ev, 3: -ev})
        r2 = fermitools.oo.omp2.twobody_amplitude_gradient(
                g[o, o, v, v], foo, fvv, t2)
        t2 += r2 / e2

        cm1oo, cm1vv = fermitools.oo.omp2.onebody_correlation_density(t2)
        m1oo = dm1oo + cm1oo
        m1vv = cm1vv
        m2oooo = fermitools.oo.omp2.twobody_moment_oooo(dm1oo, cm1oo)
        m2oovv = fermitools.oo.omp2.twobody_moment_oovv(t2)
        m2ovov = fermitools.oo.omp2.twobody_moment_ovov(dm1oo, cm1vv)

        r1 = fermitools.oo.omp2.orbital_gradient(
                h[o, v], g[o, o, o, v], g[o, v, v, v], m1oo, m1vv, m2oooo,
                m2oovv, m2ovov)
        e1 = fermitools.math.broadcast_sum({0: +eo, 1: -ev})
        t1 = r1 / e1
        gen[v, o] = numpy.transpose(t1)
        gen[o, v] = -t1
        u = scipy.linalg.expm(gen)
        c = numpy.dot(c, u)

        en_elec = fermitools.oo.omp2.electronic_energy(
                h[o, o], h[v, v], g[o, o, o, o], g[o, o, v, v], g[o, v, o, v],
                m1oo, m1vv, m2oooo, m2oovv, m2ovov)
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

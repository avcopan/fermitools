import numpy
import scipy
import warnings

import fermitools


def solve(norb, nocc, h_aso, g_aso, c_guess, t2_guess, niter=50,
          e_thresh=1e-10, r_thresh=1e-9, print_conv=False):
    no, _, nv, _ = t2_guess.shape

    zoo = numpy.zeros((no, no))
    zvv = numpy.zeros((nv, nv))

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
        fov = fermitools.oo.ocepa0.fock_xy(hov, gooov)
        fvv = fermitools.oo.ocepa0.fock_xy(hvv, govov)
        eo = numpy.diagonal(foo)
        ev = numpy.diagonal(fvv)
        e2 = fermitools.math.broadcast_sum({0: +eo, 1: +eo,
                                            2: -ev, 3: -ev})
        r2 = fermitools.oo.ocepa0.twobody_amplitude_gradient(
                goooo, goovv, govov, gvvvv, foo, fvv, t2)
        t2 += r2 / e2

        r1 = fermitools.oo.ocepa0.orbital_gradient(
                fov, gooov, govvv, t2)
        e1 = fermitools.math.broadcast_sum({0: +eo, 1: -ev})
        t1 = r1 / e1
        a = numpy.bmat([[zoo, -t1], [+t1.T, zvv]])
        u = scipy.linalg.expm(a)
        c = numpy.dot(c, u)

        en_elec = fermitools.oo.ocepa0.electronic_energy(
                hoo, hvv, goooo, goovv, govov, gvvvv, t2)
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

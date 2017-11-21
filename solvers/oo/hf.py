import numpy
import scipy.linalg
import warnings

import fermitools


def solve(norb, nocc, h_aso, g_aso, c_guess, niter=50, e_thresh=1e-10,
          r_thresh=1e-9, print_conv=False):
    o = slice(None, nocc)
    v = slice(nocc, None)

    c = c_guess
    gen = numpy.zeros((norb, norb))

    en_elec_last = 0.
    for iteration in range(niter):
        h = fermitools.math.transform(h_aso, {0: c, 1: c})
        g = fermitools.math.transform(g_aso, {0: c, 1: c, 2: c, 3: c})
        foo = fermitools.oo.hf.fock_oo(h[o, o], g[o, o, o, o])
        fvv = fermitools.oo.hf.fock_vv(h[v, v], g[o, v, o, v])

        eo = numpy.diagonal(foo)
        ev = numpy.diagonal(fvv)

        r1 = fermitools.oo.hf.orbital_gradient(h[o, v], g[o, o, o, v])
        e1 = fermitools.math.broadcast_sum({0: +eo, 1: -ev})
        t1 = r1 / e1
        gen[v, o] = numpy.transpose(t1)
        gen[o, v] = -t1
        u = scipy.linalg.expm(gen)
        c = numpy.dot(c, u)

        en_elec = fermitools.oo.hf.electronic_energy(h[o, o], g[o, o, o, o])
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


def field_energy_solver(norb, nocc, h_aso, p_aso, g_aso, c_guess, niter=50,
                        e_thresh=1e-10, r_thresh=1e-9, print_conv=False):

    def _energy(f=(0., 0., 0.)):
        hp_aso = h_aso - numpy.tensordot(f, p_aso, axes=(0, 0))
        en_elec, c = solve(norb=norb, nocc=nocc, h_aso=hp_aso,
                           g_aso=g_aso, c_guess=c_guess, niter=niter,
                           e_thresh=e_thresh, r_thresh=r_thresh,
                           print_conv=print_conv)
        return en_elec

    return _energy

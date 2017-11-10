import numpy
import scipy.linalg as spla
import functools

import warnings

import fermitools
from fermitools.math import einsum
from fermitools.math.asym import antisymmetrizer_product as asym

import interfaces.psi4 as interface
from .ocepa0 import fock
from .ocepa0 import singles_reference_density
from .ocepa0 import doubles_numerator
from .ocepa0 import doubles_cumulant
from .ocepa0 import orbital_gradient
from .ocepa0 import electronic_energy


def fancy_fock(foo, fvv, m1oo, m1vv):
    '''
    ff_ov and ff_vo are undefined, so return a dictionary with the diagonal
    blocks.
    '''
    no, uo = spla.eigh(m1oo)
    nv, uv = spla.eigh(m1vv)
    n1oo = fermitools.math.broadcast_sum({0: no, 1: no}) - 1
    n1vv = fermitools.math.broadcast_sum({0: nv, 1: nv}) - 1
    tffoo = fermitools.math.transform(foo, {0: uo, 1: uo}) / n1oo
    tffvv = fermitools.math.transform(fvv, {0: uv, 1: uv}) / n1vv
    ffoo = fermitools.math.transform(tffoo, {0: uo.T, 1: uo.T})
    ffvv = fermitools.math.transform(tffvv, {0: uv.T, 1: uv.T})
    return {'o,o': ffoo, 'v,v': ffvv}


def singles_correlation_density(t2):
    doo = -1./2 * einsum('ikcd,jkcd->ij', t2, t2)
    dvv = -1./2 * einsum('klac,klbc->ab', t2, t2)
    ioo = numpy.eye(*doo.shape)
    ivv = numpy.eye(*dvv.shape)
    m1oo = -1./2 * ioo + numpy.real(spla.sqrtm(doo + 1./4 * ioo))
    m1vv = +1./2 * ivv - numpy.real(spla.sqrtm(dvv + 1./4 * ivv))
    return spla.block_diag(m1oo, m1vv)


def doubles_density(m1, k2):
    m2 = k2 + asym("2/3")(einsum('pr,qs->pqrs', m1, m1))
    return m2


def solve(norb, nocc, h_aso, g_aso, c_guess, t2_guess, niter=50,
          e_thresh=1e-10, r_thresh=1e-9, print_conv=False):
    o = slice(None, nocc)
    v = slice(nocc, None)

    gen = numpy.zeros((norb, norb))
    m1 = m1_ref = singles_reference_density(norb=norb, nocc=nocc)

    c = c_guess
    t2_last = t2_guess
    en_elec_last = 0.
    for iteration in range(niter):
        h = fermitools.math.transform(h_aso, {0: c, 1: c})
        g = fermitools.math.transform(g_aso, {0: c, 1: c, 2: c, 3: c})
        f = fock(h, g, m1)
        ff = fancy_fock(f[o, o], f[v, v], m1[o, o], m1[v, v])

        efo = numpy.diagonal(ff['o,o'])
        efv = numpy.diagonal(ff['v,v'])
        ef2 = fermitools.math.broadcast_sum({0: +efo, 1: +efo,
                                             2: +efv, 3: +efv})
        t2 = (doubles_numerator(g[o, o, o, o], g[o, o, v, v],
                                g[o, v, o, v], g[v, v, v, v],
                                +ff['o,o'], -ff['v,v'], t2_last)
              / ef2)
        r2 = (t2 - t2_last) * ef2
        t2_last = t2
        m1_cor = singles_correlation_density(t2)
        m1 = m1_ref + m1_cor
        k2 = doubles_cumulant(t2)
        m2 = doubles_density(m1, k2)

        f = fock(h, g, m1)
        e = numpy.diagonal(f)
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
        m2 = doubles_density(m1, k2)

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


def amplitude_hessian_diag_functional(norb, nocc, h_aso, g_aso, c, step=0.01,
                                      npts=9):
    en_func = energy_functional(norb, nocc, h_aso, g_aso, c)

    def _amplitude_hessian_diag(t1_flat, t2_flat):
        en_dt2 = fermitools.math.central_difference(
                    functools.partial(en_func, t1_flat), t2_flat, step=step,
                    nder=2, npts=npts)
        return en_dt2

    return _amplitude_hessian_diag


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
    assert_almost_equal(en_tot, -74.713706346489928, decimal=10)

    # Numerically check the electronic energy gradients
    no = nocc
    nv = norb - nocc

    x = numpy.zeros(no * nv)
    t = numpy.ravel(fermitools.math.asym.compound_index(t2, {0: (0, 1),
                                                             1: (2, 3)}))
    en_dx_func = orbital_gradient_functional(norb=norb, nocc=nocc,
                                             h_aso=h_aso, g_aso=g_aso,
                                             c=c)
    en_dt_func = amplitude_gradient_functional(norb=norb, nocc=nocc,
                                               h_aso=h_aso, g_aso=g_aso,
                                               c=c)

    print("Numerical gradient calculations ...")
    en_dx = en_dx_func(x, t)
    print("... orbital gradient finished")
    en_dt = en_dt_func(x, t)
    print("... amplitude gradient finished")

    assert_almost_equal(en_dx, 0., decimal=10)
    assert_almost_equal(en_dt, 0., decimal=10)

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

    # Compare the two
    print("Compare dE/df to <Psi|mu|Psi>:")
    print(en_df.round(10))
    print(mu.round(10))
    assert_almost_equal(en_df, -mu, decimal=10)


if __name__ == '__main__':
    main()

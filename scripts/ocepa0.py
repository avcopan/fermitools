import numpy
import scipy.linalg as spla

import warnings

import fermitools
from fermitools.math.asym import antisymmetrizer_product as asym

import interfaces.psi4 as interface


def fock(hoo, hvv, goooo, govov):
    foo = hoo + numpy.trace(goooo, axis1=0, axis2=2)
    fvv = hvv + numpy.trace(govov, axis1=0, axis2=2)
    return spla.block_diag(foo, fvv)


def doubles_numerator(goooo, goovv, govov, gvvvv, foo, fvv, t2):
    foo = numpy.array(foo, copy=True)
    fvv = numpy.array(fvv, copy=True)
    numpy.fill_diagonal(foo, 0.)
    numpy.fill_diagonal(fvv, 0.)
    num2 = (goovv
            + asym("2/3")(numpy.einsum('ac,ijcb->ijab', fvv, t2))
            - asym("0/1")(numpy.einsum('ki,kjab->ijab', foo, t2))
            + 1. / 2 * numpy.einsum("abcd,ijcd->ijab", gvvvv, t2)
            + 1. / 2 * numpy.einsum("klij,klab->ijab", goooo, t2)
            - asym("0/1|2/3")(numpy.einsum("kaic,jkbc->ijab", govov, t2)))
    return num2


def singles_reference_density(norb, nocc):
    m1_ref = numpy.zeros((norb, norb))
    m1_ref[:nocc, :nocc] = numpy.eye(nocc)
    return m1_ref


def singles_correlation_density(t2):
    m1oo = - 1./2 * numpy.einsum('jkab,ikab->ij', t2, t2)
    m1vv = + 1./2 * numpy.einsum('ijac,ijbc->ab', t2, t2)
    return spla.block_diag(m1oo, m1vv)


def doubles_cumulant(t2):
    no, _, nv, _ = t2.shape
    o = slice(None, no)
    v = slice(no, None)

    n = no + nv
    k2 = numpy.zeros((n, n, n, n))
    k2[o, o, v, v] = t2
    k2[v, v, o, o] = numpy.transpose(t2)
    k2[v, v, v, v] = 1./2 * numpy.einsum('klab,klcd->abcd', t2, t2)
    k2[o, o, o, o] = 1./2 * numpy.einsum('ijcd,klcd->ijkl', t2, t2)
    k_jabi = numpy.einsum('ikac,jkbc->jabi', t2, t2)
    k2[o, v, v, o] = +numpy.transpose(k_jabi, (0, 1, 2, 3))
    k2[o, v, o, v] = -numpy.transpose(k_jabi, (0, 1, 3, 2))
    k2[v, o, v, o] = -numpy.transpose(k_jabi, (1, 0, 2, 3))
    k2[v, o, o, v] = +numpy.transpose(k_jabi, (1, 0, 3, 2))

    return k2


def doubles_density(m1_ref, m1_cor, k2):
    m2 = (k2
          + asym("0/1|2/3")(numpy.einsum('pr,qs->pqrs', m1_ref, m1_cor))
          + asym("2/3")(numpy.einsum('pr,qs->pqrs', m1_ref, m1_ref)))
    return m2


def first_order_orbital_variation_matrix(h, g, m1, m2):
    fc = (numpy.einsum('px,qx->pq', h, m1)
          + 1. / 2 * numpy.einsum('pxyz,qxyz->pq', g, m2))
    return fc


def singles_residual(o, v, h, g, m1, m2):
    fc = first_order_orbital_variation_matrix(h, g, m1, m2)
    res1 = (fc - numpy.transpose(fc))[o, v]
    return res1


def electronic_energy(h, g, m1, m2):
    return numpy.vdot(h, m1) + 1. / 4 * numpy.vdot(g, m2)


def solve_ocepa0(norb, nocc, h_aso, g_aso, c_guess, t2_guess, niter=50,
                 e_thresh=1e-10, r_thresh=1e-9, print_conv=False):
    o = slice(None, nocc)
    v = slice(nocc, None)

    x1 = numpy.zeros((norb, norb))
    m1_ref = singles_reference_density(norb=norb, nocc=nocc)

    c = c_guess
    t2_last = t2_guess
    en_elec_last = 0.
    for iteration in range(niter):
        h = fermitools.math.transform(h_aso, {0: c, 1: c})
        g = fermitools.math.transform(g_aso, {0: c, 1: c, 2: c, 3: c})
        f = fock(h[o, o], h[v, v], g[o, o, o, o], g[o, v, o, v])

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

        r1 = singles_residual(o, v, h, g, m1, m2)
        e1 = fermitools.math.broadcast_sum({0: +e[o], 1: -e[v]})
        t1 = r1 / e1
        x1[o, v] = t1
        u = spla.expm(x1 - numpy.transpose(x1))
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


def electronic_energy_functional(norb, nocc, h_aso, g_aso, c):
    o = slice(None, nocc)
    v = slice(nocc, None)

    m1_ref = singles_reference_density(norb=norb, nocc=nocc)

    import itertools as it

    def electronic_energy_function(t1, t2_flat):

        # hack -- generalize this at some point
        nvir = norb - nocc
        t2 = numpy.zeros((nocc, nocc, nvir, nvir))
        ijab_iterator = it.product(it.combinations(range(nocc), 2),
                                   it.combinations(range(nvir), 2))
        for ijab, ((i, j), (a, b)) in enumerate(ijab_iterator):
            t2[i, j, a, b] = t2_flat[ijab]

        t2 = asym('0/1|2/3')(t2)

        x1 = numpy.zeros((norb, norb))
        x1[o, v] = t1
        u = spla.expm(x1 - numpy.transpose(x1))
        ct = numpy.dot(c, u)

        h = fermitools.math.transform(h_aso, {0: ct, 1: ct})
        g = fermitools.math.transform(g_aso, {0: ct, 1: ct, 2: ct, 3: ct})

        m1_cor = singles_correlation_density(t2)
        m1 = m1_ref + m1_cor
        k2 = doubles_cumulant(t2)
        m2 = doubles_density(m1_ref, m1_cor, k2)

        return electronic_energy(h, g, m1, m2)

    return electronic_energy_function


def perturbed_energy_function(norb, nocc, h_aso, p_aso, g_aso, c_guess,
                              t2_guess, niter=50, e_thresh=1e-10,
                              r_thresh=1e-9, print_conv=False):

    def electronic_energy(f=(0., 0., 0.)):
        hp_aso = h_aso - numpy.tensordot(f, p_aso, axes=(0, 0))
        en_elec, c, t2 = solve_ocepa0(norb=norb, nocc=nocc, h_aso=hp_aso,
                                      g_aso=g_aso, c_guess=c_guess,
                                      t2_guess=t2_guess, niter=niter,
                                      e_thresh=e_thresh, r_thresh=r_thresh,
                                      print_conv=print_conv)
        return en_elec

    return electronic_energy


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
    c_unsrt = spla.block_diag(ac, bc)
    sortvec = fermitools.math.spinorb.ab2ov(dim=nbf, na=na, nb=nb)
    c_unsrt = spla.block_diag(ac, bc)
    c = fermitools.math.spinorb.sort(c_unsrt, order=sortvec, axes=(1,))

    en_nuc = fermitools.chem.nuc.energy(labels=LABELS, coords=COORDS)

    t2_guess = numpy.zeros((nocc, nocc, norb-nocc, norb-nocc))
    en_elec, c, t2 = solve_ocepa0(norb=norb, nocc=nocc, h_aso=h_aso,
                                  g_aso=g_aso, c_guess=c, t2_guess=t2_guess,
                                  niter=200, e_thresh=1e-14, r_thresh=1e-12,
                                  print_conv=True)
    en_tot = en_elec + en_nuc
    print(en_tot)

    # Evaluate dipole moment as expectation value
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

    print(en_df.round(10))
    print(mu.round(10))

    print('{:20.15f}'.format(en_tot))

    from numpy.testing import assert_almost_equal
    assert_almost_equal(en_tot, -74.71451994543345, decimal=10)
    assert_almost_equal(en_df, -mu, decimal=11)


if __name__ == '__main__':
    main()

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


def singles_density(norb, nocc):
    m1 = numpy.zeros((norb, norb))
    m1[:nocc, :nocc] = numpy.eye(nocc)
    return m1


def doubles_density(m1):
    m2 = asym("2/3")(numpy.einsum('pr,qs->pqrs', m1, m1))
    return m2


def first_order_orbital_variation_matrix(h, g, m1, m2):
    fc = (numpy.einsum('px,qx->pq', h, m1)
          + 1. / 2 * numpy.einsum('pxyz,qxyz->pq', g, m2))
    return fc


def orbital_gradient(o, v, h, g, m1, m2):
    fc = first_order_orbital_variation_matrix(h, g, m1, m2)
    res1 = (fc - numpy.transpose(fc))[o, v]
    return res1


def electronic_energy(h, g, m1, m2):
    return numpy.vdot(h, m1) + 1. / 4 * numpy.vdot(g, m2)


def electronic_energy_functional(norb, nocc, h_aso, g_aso, c):
    o = slice(None, nocc)
    v = slice(nocc, None)

    m1 = singles_density(norb=norb, nocc=nocc)
    m2 = doubles_density(m1)

    def electronic_energy_function(t1):
        x1 = numpy.zeros((norb, norb))
        x1[o, v] = t1
        u = spla.expm(x1 - numpy.transpose(x1))
        ct = numpy.dot(c, u)

        h = fermitools.math.transform(h_aso, {0: ct, 1: ct})
        g = fermitools.math.transform(g_aso, {0: ct, 1: ct, 2: ct, 3: ct})

        return electronic_energy(h, g, m1, m2)

    return electronic_energy_function


def solve_ohf(norb, nocc, h_aso, g_aso, c_guess, niter=50, e_thresh=1e-10,
              r_thresh=1e-9, print_conv=False):
    o = slice(None, nocc)
    v = slice(nocc, None)

    c = c_guess
    x1 = numpy.zeros((norb, norb))

    m1 = singles_density(norb=norb, nocc=nocc)
    m2 = doubles_density(m1)

    en_elec_last = 0.
    for iteration in range(niter):
        h = fermitools.math.transform(h_aso, {0: c, 1: c})
        g = fermitools.math.transform(g_aso, {0: c, 1: c, 2: c, 3: c})
        f = fock(h[o, o], h[v, v], g[o, o, o, o], g[o, v, o, v])

        e = numpy.diagonal(f)

        r1 = orbital_gradient(o, v, h, g, m1, m2)
        e1 = fermitools.math.broadcast_sum({0: +e[o], 1: -e[v]})
        t1 = r1 / e1
        x1[o, v] = t1
        u = spla.expm(x1 - numpy.transpose(x1))
        c = numpy.dot(c, u)

        en_elec = electronic_energy(h, g, m1, m2)
        en_change = en_elec - en_elec_last
        en_elec_last = en_elec

        r_norm = spla.norm(r1)

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

    def electronic_energy(f=(0., 0., 0.)):
        hp_aso = h_aso - numpy.tensordot(f, p_aso, axes=(0, 0))
        en_elec, c = solve_ohf(norb=norb, nocc=nocc, h_aso=hp_aso,
                               g_aso=g_aso, c_guess=c_guess, niter=niter,
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
    en_elec, c = solve_ohf(norb=norb, nocc=nocc, h_aso=h_aso, g_aso=g_aso,
                           c_guess=c, niter=200, e_thresh=1e-14,
                           r_thresh=1e-12, print_conv=True)
    en_tot = en_elec + en_nuc
    print(en_tot)

    # Evaluate dipole moment as expectation value
    p = fermitools.math.transform(p_aso, {1: c, 2: c})
    m1 = singles_density(norb=norb, nocc=nocc)
    mu = numpy.array([numpy.vdot(px, m1) for px in p])

    # Evaluate dipole moment as energy derivative
    en_f = perturbed_energy_function(norb=norb, nocc=nocc, h_aso=h_aso,
                                     p_aso=p_aso, g_aso=g_aso, c_guess=c,
                                     niter=200, e_thresh=1e-13, r_thresh=1e-9,
                                     print_conv=True)
    en_df = fermitools.math.central_difference(en_f, (0., 0., 0.),
                                               step=0.002, npts=9)

    print(en_df.round(10))
    print(mu.round(10))

    print('{:20.15f}'.format(en_tot))

    from numpy.testing import assert_almost_equal
    assert_almost_equal(en_tot, -74.66178436045595, decimal=10)
    assert_almost_equal(en_df, -mu, decimal=11)


if __name__ == '__main__':
    main()

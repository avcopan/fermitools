import numpy
import scipy.linalg as spla

import fermitools

import interfaces.psi4 as interface
from . import ohf


def second_order_orbital_variation_tensor(h, g, m1, m2):
    i = numpy.eye(*h.shape)
    fc = ohf.first_order_orbital_variation_matrix(h, g, m1, m2)
    hc = (+ numpy.einsum('ps,qr->prqs', i, fc)
          + numpy.einsum('ps,qr->prqs', fc.T, i)
          - numpy.einsum('ps,qr->prqs', h, m1)
          - numpy.einsum('ps,qr->prqs', m1, h)
          - numpy.einsum('pysx,qyrx->prqs', g, m2)
          - numpy.einsum('pysx,qyrx->prqs', m2, g)
          + 1. / 2 * numpy.einsum('prxy,qsxy->prqs', g, m2)
          + 1. / 2 * numpy.einsum('prxy,qsxy->prqs', m2, g))
    return hc


def diagonal_orbital_hessian(nocc, norb, h, g, m1, m2):
    o = slice(None, nocc)
    v = slice(nocc, None)
    no = nocc
    nv = norb - nocc
    hc = second_order_orbital_variation_tensor(h, g, m1, m2)
    a = -1./2 * (numpy.einsum('ibaj->iajb', hc[o, v, v, o]) +
                 numpy.einsum('bija->iajb', hc[v, o, o, v]))
    return numpy.reshape(a, (no * nv, no * nv))


def offdiagonal_orbital_hessian(nocc, norb, h, g, m1, m2):
    o = slice(None, nocc)
    v = slice(nocc, None)
    no = nocc
    nv = norb - nocc
    hc = second_order_orbital_variation_tensor(h, g, m1, m2)
    a = 1./2 * (numpy.einsum('ijab->iajb', hc[o, o, v, v]) +
                numpy.einsum('jiba->iajb', hc[o, o, v, v]))
    return numpy.reshape(a, (no * nv, no * nv))


def orbital_property_gradient(o, v, p, m1):
    t = (numpy.dot(p, m1) - numpy.dot(m1, p))[o, v]
    return numpy.ravel(t)


def static_response_vector(a, b, t):
    """solve for the static response vector

    :param a: diagonal orbital hessian
    :type a: numpy.ndarray
    :param b: off-diagonal orbital hessian
    :type b: numpy.ndarray
    :param t: property gradient vector(s)
    :type t: numpy.ndarray
    :returns: the response vector(s), (x + y) = 2 * (a + b)^-1 * t
    :rtype: numpy.ndarray
    """
    return spla.solve(a + b, 2 * t, sym_pos=True)


def static_linear_response_function(t, r):
    """the linear response function, evaluated at zero field strength (=static)

    :param t: property gradient vector(s)
    :type t: numpy.ndarray
    :param r: the response vector(s), (x + y) = 2 * (a + b)^-1 * t
    :type r: numpy.ndarray
    :returns: the response function(s)
    :rtype: float or numpy.ndarray
    """
    return numpy.dot(numpy.transpose(t), r)


def spectrum(a, b):
    w2 = spla.eigvals(numpy.dot(a + b, a - b))
    return numpy.array(sorted(numpy.sqrt(w2.real)))


def tamm_dancoff_spectrum(a):
    return spla.eigvalsh(a)


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
    en_elec, c = ohf.solve_ohf(norb=norb, nocc=nocc, h_aso=h_aso, g_aso=g_aso,
                               c_guess=c, niter=200, e_thresh=1e-14,
                               r_thresh=1e-11, print_conv=200)
    en_tot = en_elec + en_nuc
    print(en_tot)

    # Build the orbital Hessian w/ general formula
    h = fermitools.math.transform(h_aso, {0: c, 1: c})
    g = fermitools.math.transform(g_aso, {0: c, 1: c, 2: c, 3: c})
    m1 = ohf.singles_density(norb=norb, nocc=nocc)
    m2 = ohf.doubles_density(m1)
    a_orb = diagonal_orbital_hessian(nocc, norb, h, g, m1, m2)

    # Build the orbital Hessian w/ RPA formula
    from . import rpa

    o = slice(None, nocc)
    v = slice(nocc, None)

    f = ohf.fock(h[o, o], h[v, v], g[o, o, o, o], g[o, v, o, v])
    a_orb_rpa = rpa.diagonal_orbital_hessian(g[o, v, o, v], f[o, o], f[v, v])

    # Test derivatives
    from numpy.testing import assert_almost_equal

    t1 = numpy.zeros((nocc, norb-nocc))
    en_orb = ohf.electronic_energy_functional(norb=norb, nocc=nocc,
                                              h_aso=h_aso, g_aso=g_aso, c=c)

    en_dorb = numpy.ravel(fermitools.math.central_difference(en_orb, t1,
                                                             step=0.05,
                                                             nder=1,
                                                             npts=23))
    print(en_dorb.round(10))

    assert_almost_equal(en_dorb, 0., decimal=10)

    en_dorb2 = numpy.ravel(fermitools.math.central_difference(en_orb, t1,
                                                              step=0.05,
                                                              nder=2,
                                                              npts=23))
    print("Orbital Hessian:")
    print((numpy.diag(a_orb) - en_dorb2 / 2.).round(10))
    print((numpy.diag(a_orb_rpa) - en_dorb2 / 2.).round(10))

    assert_almost_equal(numpy.diag(a_orb), en_dorb2 / 2., decimal=10)
    assert_almost_equal(numpy.diag(a_orb_rpa), en_dorb2 / 2., decimal=10)


if __name__ == '__main__':
    main()

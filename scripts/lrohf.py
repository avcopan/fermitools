import numpy
import scipy.linalg as spla

import fermitools

import interfaces.psi4 as interface
from . import ohf


def second_order_orbital_variation_tensor(h, g, m1, m2):
    i = numpy.eye(*h.shape)
    fc = ohf.first_order_orbital_variation_matrix(h, g, m1, m2)
    a = (+ numpy.einsum('uv,tw->tuvw', i, fc + fc.T) / 2.
         + numpy.einsum('tw,uv->tuvw', i, fc + fc.T) / 2.
         - numpy.einsum('tv,uw->tuvw', i, fc + fc.T) / 2.
         - numpy.einsum('uw,vt->tuvw', i, fc + fc.T) / 2.
         + numpy.einsum('tv,wu->tuvw', h, m1)
         - numpy.einsum('uv,wt->tuvw', h, m1)
         - numpy.einsum('tw,vu->tuvw', h, m1)
         + numpy.einsum('uw,vt->tuvw', h, m1)
         + numpy.einsum('pqvt,pqwu->tuvw', g, m2) / 2.
         - numpy.einsum('pqvu,pqwt->tuvw', g, m2) / 2.
         - numpy.einsum('pqwt,pqvu->tuvw', g, m2) / 2.
         + numpy.einsum('pqwu,pqvt->tuvw', g, m2) / 2.
         + numpy.einsum('pvqt,pwqu->tuvw', g, m2)
         - numpy.einsum('pvqu,pwqt->tuvw', g, m2)
         - numpy.einsum('pwqt,pvqu->tuvw', g, m2)
         + numpy.einsum('pwqu,pvqt->tuvw', g, m2))
    return a


def diagonal_orbital_hessian(nocc, norb, h, g, m1, m2):
    o = slice(None, nocc)
    v = slice(nocc, None)
    no = nocc
    nv = norb - nocc
    h = second_order_orbital_variation_tensor(h, g, m1, m2)
    a = h[o, v, o, v]
    return numpy.reshape(a, (no * nv, no * nv))


def offdiagonal_orbital_hessian(nocc, norb, h, g, m1, m2):
    o = slice(None, nocc)
    v = slice(nocc, None)
    no = nocc
    nv = norb - nocc
    h = second_order_orbital_variation_tensor(h, g, m1, m2)
    b = numpy.transpose(-h[o, v, v, o], (0, 1, 3, 2))
    return numpy.reshape(b, (no * nv, no * nv))


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
    p = fermitools.math.transform(p_aso, {1: c, 2: c})
    g = fermitools.math.transform(g_aso, {0: c, 1: c, 2: c, 3: c})
    m1 = ohf.singles_density(norb=norb, nocc=nocc)
    m2 = ohf.doubles_density(m1)
    a_orb = diagonal_orbital_hessian(nocc, norb, h, g, m1, m2)
    b_orb = offdiagonal_orbital_hessian(nocc, norb, h, g, m1, m2)

    o = slice(None, nocc)
    v = slice(nocc, None)
    t_orb = numpy.transpose(
        [orbital_property_gradient(o, v, px, m1) for px in p])

    r_orb = static_response_vector(a_orb, b_orb, t_orb)
    alpha = static_linear_response_function(t_orb, r_orb)

    # Evaluate dipole polarizability as energy derivative
    en_f = ohf.perturbed_energy_function(norb=norb, nocc=nocc, h_aso=h_aso,
                                         p_aso=p_aso, g_aso=g_aso,
                                         c_guess=c, niter=200,
                                         e_thresh=1e-14, r_thresh=1e-11,
                                         print_conv=True)
    en_df2 = fermitools.math.central_difference(en_f, (0., 0., 0.),
                                                step=0.007, nder=2, npts=7)
    print(en_df2.round(10)/2.)
    print(numpy.diag(alpha).round(10))

    from numpy.testing import assert_almost_equal

    assert_almost_equal(numpy.diag(alpha), -en_df2/2., decimal=9)

    # Build the orbital Hessian w/ RPA formula
    from . import rpa

    o = slice(None, nocc)
    v = slice(nocc, None)

    f = ohf.fock(h[o, o], h[v, v], g[o, o, o, o], g[o, v, o, v])
    a_orb_rpa = rpa.diagonal_orbital_hessian(g[o, v, o, v], f[o, o], f[v, v])

    # Test derivatives
    t1 = numpy.zeros((nocc, norb-nocc))
    en_orb = ohf.electronic_energy_functional(norb=norb, nocc=nocc,
                                              h_aso=h_aso, g_aso=g_aso, c=c)

    en_dorb = numpy.ravel(fermitools.math.central_difference(en_orb, t1,
                                                             step=0.05,
                                                             nder=1,
                                                             npts=9))
    print(en_dorb.round(10))

    assert_almost_equal(en_dorb, 0., decimal=10)

    en_dorb2 = numpy.ravel(fermitools.math.central_difference(en_orb, t1,
                                                              step=0.05,
                                                              nder=2,
                                                              npts=9))
    print("Orbital Hessian:")
    print((numpy.diag(a_orb) - en_dorb2 / 2.).round(9))
    print((numpy.diag(a_orb_rpa) - en_dorb2 / 2.).round(9))

    assert_almost_equal(numpy.diag(a_orb), en_dorb2 / 2., decimal=10)
    assert_almost_equal(numpy.diag(a_orb_rpa), en_dorb2 / 2., decimal=10)


if __name__ == '__main__':
    main()

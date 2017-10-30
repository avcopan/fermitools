import numpy
import scipy.linalg as spla

import fermitools

import interfaces.psi4 as interface
from . import scf


def second_order_orbital_variation_tensor(h, g, m1, m2):
    i = numpy.eye(*h.shape)
    fc = scf.first_order_orbital_variation_matrix(h, g, m1, m2)
    fcs = (fc + numpy.transpose(fc)) / 2.
    hc = (+ numpy.einsum('pr,qs->pqrs', h, m1)
          + numpy.einsum('pr,qs->pqrs', m1, h)
          - numpy.einsum('pr,qs->pqrs', i, fcs)
          - numpy.einsum('pr,qs->pqrs', fcs, i)
          + numpy.einsum('pxry,qxsy->pqrs', g, m2)
          + numpy.einsum('pxry,qxsy->pqrs', m2, g)
          - 1./2. * numpy.einsum('psxy,qrxy->pqrs', g, m2)
          - 1./2. * numpy.einsum('psxy,qrxy->pqrs', m2, g))
    return hc


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
    b = -numpy.transpose(h[o, v, v, o], (0, 1, 3, 2))
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
    CHARGE = 0
    SPIN = 0
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
    en_elec, c = scf.solve_scf(norb=norb, nocc=nocc, h_aso=h_aso, g_aso=g_aso,
                               c_guess=c, niter=200, e_thresh=1e-14,
                               r_thresh=1e-12, print_conv=200)
    en_tot = en_elec + en_nuc
    print("Total energy:")
    print('{:20.15f}'.format(en_tot))

    # Evalute the dipole polarizability as a linear response function
    h = fermitools.math.transform(h_aso, {0: c, 1: c})
    g = fermitools.math.transform(g_aso, {0: c, 1: c, 2: c, 3: c})
    m1 = scf.singles_density(norb=norb, nocc=nocc)
    m2 = scf.doubles_density(m1)
    a_orb = diagonal_orbital_hessian(nocc, norb, h, g, m1, m2)
    b_orb = offdiagonal_orbital_hessian(nocc, norb, h, g, m1, m2)

    w_rpa = spectrum(a_orb, b_orb)

    from numpy.testing import assert_almost_equal
    w_rpa_ref = [0.2851637170, 0.2851637170, 0.2851637170, 0.2997434467,
                 0.2997434467, 0.2997434467, 0.3526266606, 0.3526266606,
                 0.3526266606, 0.3547782530, 0.3651313107, 0.3651313107,
                 0.3651313107, 0.4153174946, 0.5001011401, 0.5106610509,
                 0.5106610509, 0.5106610509, 0.5460719086, 0.5460719086,
                 0.5460719086, 0.5513718846, 0.6502707118, 0.8734253708,
                 1.1038187957, 1.1038187957, 1.1038187957, 1.1957870714,
                 1.1957870714, 1.1957870714, 1.2832053178, 1.3237421886,
                 19.9585040647, 19.9585040647, 19.9585040647, 20.0109471551,
                 20.0113074586, 20.0113074586, 20.0113074586, 20.0504919449]

    assert_almost_equal(w_rpa, w_rpa_ref, decimal=10)

    # Test derivatives
    import os

    no = nocc
    nv = norb - nocc
    x = numpy.zeros(no * nv)

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'data')

    en_dxdx_func = scf.orbital_hessian_functional(norb=norb, nocc=nocc,
                                                  h_aso=h_aso, g_aso=g_aso,
                                                  c=c, step=0.05, npts=11)

    def generate_orbital_hessian():
        en_dxdx = en_dxdx_func(x)
        numpy.save(os.path.join(data_path, 'lr_scf/en_dxdx.npy'), en_dxdx)

    # generate_orbital_hessian()
    en_dxdx = numpy.load(os.path.join(data_path, 'lr_scf/en_dxdx.npy'))

    print(numpy.diag(en_dxdx).round(9))
    print(numpy.diag(a_orb + b_orb).round(9))
    print((numpy.diag(a_orb + b_orb) / numpy.diag(en_dxdx)).round(9))
    print((en_dxdx - 2*(a_orb + b_orb)).round(8))
    print(spla.norm(en_dxdx - 2*(a_orb + b_orb)))


if __name__ == '__main__':
    main()

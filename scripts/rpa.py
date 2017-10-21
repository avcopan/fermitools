import numpy
import scipy.linalg as spla

import fermitools

import interfaces.psi4 as interface


def diagonal_orbital_hessian(govov, foo, fvv):
    no, nv, _, _ = govov.shape
    ioo = numpy.eye(no)
    ivv = numpy.eye(nv)
    a = (+ numpy.einsum('ab,ij->iajb', fvv, ioo)
         - numpy.einsum('ij,ab->iajb', foo, ivv)
         - numpy.einsum('ibja->iajb', govov))
    return numpy.reshape(a, (no * nv, no * nv))


def offdiagonal_orbital_hessian(goovv):
    no, _, nv, _ = goovv.shape
    b = numpy.transpose(goovv, (0, 2, 1, 3))
    return numpy.reshape(b, (no * nv, no * nv))


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


def tamm_dancoff_spectrum(a):
    return spla.eigvalsh(a)


def spectrum(a, b):
    w2 = spla.eigvals(numpy.dot(a + b, a - b))
    return numpy.array(sorted(numpy.sqrt(w2.real)))


def driver(basis, labels, coords, charge, spin):
    # Spaces
    nbf = interface.integrals.nbf(basis, labels)
    na = fermitools.chem.elec.count_alpha(labels, charge, spin)
    nb = fermitools.chem.elec.count_beta(labels, charge, spin)
    n_elec = na + nb
    o = slice(None, n_elec)
    v = slice(n_elec, None)

    # Integrals
    h_ao = interface.integrals.core_hamiltonian(basis, labels, coords)
    p_ao = interface.integrals.dipole(basis, labels, coords)
    r_ao = interface.integrals.repulsion(basis, labels, coords)

    h_aso = fermitools.math.spinorb.expand(h_ao, brakets=((0, 1),))
    p_aso = fermitools.math.spinorb.expand(p_ao, brakets=((1, 2),))
    r_aso = fermitools.math.spinorb.expand(r_ao, brakets=((0, 2), (1, 3)))
    g_aso = r_aso - numpy.transpose(r_aso, (0, 1, 3, 2))

    # MO coefficients
    from fermitools.math.spinorb import ab2ov
    ac, bc = interface.hf.unrestricted_orbitals(basis, labels, coords,
                                                charge, spin)
    c_unsrt = spla.block_diag(ac, bc)
    c = fermitools.math.spinorb.sort(c_unsrt,
                                     order=ab2ov(dim=nbf, na=na, nb=nb),
                                     axes=(1,))
    co = c[:, o]
    cv = c[:, v]

    # MO basis integrals
    d_aso = fermitools.scf.density(co)
    f_aso = fermitools.scf.hf.fock(h=h_aso, g=g_aso, d=d_aso)

    foo = fermitools.math.transform(f_aso, {0: co, 1: co})
    fvv = fermitools.math.transform(f_aso, {0: cv, 1: cv})
    pov = fermitools.math.transform(p_aso, {1: co, 2: cv})
    govov = fermitools.math.transform(g_aso, {0: co, 1: cv, 2: co, 3: cv})
    goovv = fermitools.math.transform(g_aso, {0: co, 1: co, 2: cv, 3: cv})

    a = diagonal_orbital_hessian(govov, foo, fvv)
    b = offdiagonal_orbital_hessian(goovv)

    w_td = tamm_dancoff_spectrum(a)
    print(w_td)
    w_rpa = spectrum(a, b)
    print(w_rpa)

    t = numpy.transpose(numpy.reshape(pov, (3, -1)))
    r = static_response_vector(a, b, t)
    alpha = static_linear_response_function(t, r)

    print(alpha.round(10))

    from numpy.testing import assert_almost_equal

    w_td_ref = [0.2872554996, 0.2872554996, 0.2872554996, 0.3444249963,
                0.3444249963, 0.3444249963, 0.3564617587, 0.3659889948,
                0.3659889948, 0.3659889948, 0.3945137992, 0.3945137992,
                0.3945137992, 0.4160717386, 0.5056282877, 0.5142899971,
                0.5142899971, 0.5142899971, 0.5551918860, 0.5630557635,
                0.5630557635, 0.5630557635, 0.6553184485, 0.9101216891,
                1.1087709658, 1.1087709658, 1.1087709658, 1.2000961331,
                1.2000961331, 1.2000961331, 1.3007851948, 1.3257620652,
                19.9585264123, 19.9585264123, 19.9585264123, 20.0109794203,
                20.0113420895, 20.0113420895, 20.0113420895, 20.0505319444]

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

    assert_almost_equal(w_td, w_td_ref, decimal=10)
    assert_almost_equal(w_rpa, w_rpa_ref, decimal=10)

    # Test the response function by comparing the dipole polarizability
    # matrix to the electric field Hessian
    from . import uhf

    en_f = uhf.energy_function(basis=basis, labels=labels, coords=coords,
                               charge=charge, spin=spin, e_thresh=1e-14)
    en_df2 = fermitools.math.central_difference(en_f, (0., 0., 0.),
                                                step=0.005, nder=2, npts=13)
    assert_almost_equal(numpy.diag(alpha), -en_df2, decimal=8)


def main():
    CHARGE = 0
    SPIN = 0
    BASIS = 'sto-3g'
    LABELS = ('O', 'H', 'H')
    COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
              (0.000000000000,  1.638036840407,  1.136548822547),
              (0.000000000000, -1.638036840407,  1.136548822547))

    driver(basis=BASIS, labels=LABELS, coords=COORDS, charge=CHARGE,
           spin=SPIN)


if __name__ == '__main__':
    main()

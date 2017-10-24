import numpy
import scipy.linalg as spla

import fermitools
from fermitools.math.asym import antisymmetrizer_product as asym

import interfaces.psi4 as interface
from . import ocepa0


def second_order_orbital_variation_tensor(h, g, m1, m2):
    i = numpy.eye(*h.shape)
    fc = ocepa0.first_order_orbital_variation_matrix(h, g, m1, m2)
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


def diagonal_amplitude_hessian(foo, fvv, goooo, govov, gvvvv):
    no, nv, _, _ = govov.shape
    ndoubles = no * (no - 1) * nv * (nv - 1) // 4
    io = numpy.eye(*foo.shape)
    iv = numpy.eye(*fvv.shape)
    a = (+ asym('2/3|4/5|6/7')(
               numpy.einsum('ik,jl,ac,bd->ijabklcd', io, io, fvv, iv))
         - asym('0/1|4/5|6/7')(
               numpy.einsum('ik,jl,ac,bd->ijabklcd', foo, io, iv, iv))
         + asym('4/5')(
               numpy.einsum('ik,jl,abcd->ijabklcd', io, io, gvvvv))
         + asym('6/7')(
               numpy.einsum('ijkl,ac,bd->ijabklcd', goooo, iv, iv))
         - asym('0/1|2/3|4/5|6/7')(
               numpy.einsum('ik,jcla,bd->ijabklcd', io, govov, iv)))
    a_cmp = fermitools.math.asym.compound_index(a, {0: (0, 1), 1: (2, 3),
                                                    2: (4, 5), 3: (6, 7)})
    return numpy.reshape(a_cmp, (ndoubles, ndoubles))


def diagonal_mixed_hessian(o, v, g, t2):
    no, _, nv, _ = t2.shape
    nsingles = no * nv
    ndoubles = no * (no - 1) * nv * (nv - 1) // 4
    io = numpy.eye(no)
    iv = numpy.eye(nv)
    a = (+ asym('0/1')(
               numpy.einsum('abic,jk->ijabkc', g[v, v, o, v], io))
         - asym('2/3')(
               numpy.einsum('akij,bc->ijabkc', g[v, o, o, o], iv))
         + asym('0/1|2/3')(
               numpy.einsum('imae,mbec,jk->ijabkc', t2, g[o, v, v, v], io))
         - asym('0/1|2/3')(
               numpy.einsum('imae,mkej,bc->ijabkc', t2, g[o, o, v, o], iv))
         - 1./2 * asym('0/1')(
               numpy.einsum('mnab,mnic,jk->ijabkc', t2, g[o, o, o, v], io))
         + 1./2 * asym('2/3')(
               numpy.einsum('ijef,akef,bc->ijabkc', t2, g[v, o, v, v], iv))
         - asym('0/1')(
               numpy.einsum('imab,mkjc->ijabkc', t2, g[o, o, o, v]))
         + asym('2/3')(
               numpy.einsum('ijae,bkec->ijabkc', t2, g[v, o, v, v])))
    a_cmp = fermitools.math.asym.compound_index(a, {0: (0, 1), 1: (2, 3)})
    return numpy.reshape(a_cmp, (ndoubles, nsingles))


def offdiagonal_mixed_hessian(o, v, g, t2):
    no, _, nv, _ = t2.shape
    nsingles = no * nv
    ndoubles = no * (no - 1) * nv * (nv - 1) // 4
    b = (- numpy.einsum('ijec,abek->ijabkc', t2, g[v, v, v, o])
         + numpy.einsum('mkab,mcij->ijabkc', t2, g[o, v, o, o])
         + asym('0/1')(
               numpy.einsum('imab,mcjk->ijabkc', t2, g[o, v, o, o]))
         - asym('2/3')(
               numpy.einsum('ijae,bcek->ijabkc', t2, g[v, v, v, o]))
         + asym('0/1|2/3')(
               numpy.einsum('ikae,bcje->ijabkc', t2, g[v, v, o, v]))
         - asym('0/1|2/3')(
               numpy.einsum('imac,bmjk->ijabkc', t2, g[v, o, o, o])))
    b_cmp = fermitools.math.asym.compound_index(b, {0: (0, 1), 1: (2, 3)})
    return numpy.reshape(b_cmp, (ndoubles, nsingles))


def orbital_property_gradient(o, v, p, m1):
    t = (numpy.dot(p, m1) - numpy.dot(m1, p))[o, v]
    return numpy.ravel(t)


def amplitude_property_gradient(poo, pvv, t2):
    no, _, nv, _ = t2.shape
    t = (+ asym('2/3')(
               numpy.einsum('ac,ijcb->ijab', pvv, t2))
         - asym('0/1')(
               numpy.einsum('ik,kjab->ijab', poo, t2)))
    t_cmp = fermitools.math.asym.compound_index(t, {0: (0, 1), 1: (2, 3)})
    return numpy.ravel(t_cmp)


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


def driver(basis, labels, coords, charge, spin):
    na = fermitools.chem.elec.count_alpha(labels, charge, spin)
    nb = fermitools.chem.elec.count_beta(labels, charge, spin)
    nocc = na + nb
    o = slice(None, nocc)
    v = slice(nocc, None)

    ac, bc = interface.hf.unrestricted_orbitals(basis, labels, coords,
                                                charge, spin)
    nbf = interface.integrals.nbf(basis, labels)
    norb = 2 * nbf
    h_ao = interface.integrals.core_hamiltonian(basis, labels, coords)
    p_ao = interface.integrals.dipole(basis, labels, coords)
    r_ao = interface.integrals.repulsion(basis, labels, coords)

    h_aso = fermitools.math.spinorb.expand(h_ao, brakets=((0, 1),))
    p_aso = fermitools.math.spinorb.expand(p_ao, brakets=((1, 2),))
    r_aso = fermitools.math.spinorb.expand(r_ao, brakets=((0, 2), (1, 3)))
    g_aso = r_aso - numpy.transpose(r_aso, (0, 1, 3, 2))

    from fermitools.math.spinorb import ab2ov

    c_unsrt = spla.block_diag(ac, bc)
    c = fermitools.math.spinorb.sort(c_unsrt,
                                     order=ab2ov(dim=nbf, na=na, nb=nb),
                                     axes=(1,))

    en_elec, c, t2 = ocepa0.solve_ocepa0(norb=2*nbf, nocc=na+nb, h_aso=h_aso,
                                         g_aso=g_aso, c_guess=c, niter=200,
                                         e_thresh=1e-14, r_thresh=1e-10,
                                         print_conv=200)
    h = fermitools.math.transform(h_aso, {0: c, 1: c})
    p = fermitools.math.transform(p_aso, {1: c, 2: c})
    g = fermitools.math.transform(g_aso, {0: c, 1: c, 2: c, 3: c})
    f = ocepa0.fock(h[o, o], h[v, v], g[o, o, o, o], g[o, v, o, v])
    m1_ref = ocepa0.singles_reference_density(norb=norb, nocc=nocc)
    m1_cor = ocepa0.singles_correlation_density(t2)
    m1 = m1_ref + m1_cor
    k2 = ocepa0.doubles_cumulant(t2)
    m2 = ocepa0.doubles_density(m1_ref, m1_cor, k2)

    a_orb = diagonal_orbital_hessian(nocc, norb, h, g, m1, m2)
    a_mix = diagonal_mixed_hessian(o, v, g, t2)
    a_amp = diagonal_amplitude_hessian(f[o, o], f[v, v], g[o, o, o, o],
                                       g[o, v, o, v], g[v, v, v, v])

    b_orb = offdiagonal_orbital_hessian(nocc, norb, h, g, m1, m2)
    b_mix = offdiagonal_mixed_hessian(o, v, g, t2)
    b_amp = numpy.zeros_like(a_amp)

    t_orb = numpy.transpose(
        [orbital_property_gradient(o, v, px, m1) for px in p])
    t_amp = numpy.transpose(
        [amplitude_property_gradient(px[o, o], px[v, v], t2) for px in p])

    a = numpy.bmat([[a_orb, a_mix.T], [a_mix, a_amp]])
    b = numpy.bmat([[b_orb, b_mix.T], [b_mix, b_amp]])
    t = numpy.bmat([[t_orb], [t_amp]])

    r = static_response_vector(a, b, t)
    alpha = static_linear_response_function(t, r)

    # Evaluate dipole polarizability as energy derivative
    en_f = ocepa0.perturbed_energy_function(basis, labels, coords, charge,
                                            spin, niter=200, e_thresh=1e-11,
                                            r_thresh=1e-9, print_conv=True)
    en_df2 = fermitools.math.central_difference(en_f, (0., 0., 0.),
                                                step=0.002, nder=2, npts=9)
    print(en_df2.round(10))
    print(numpy.diag(alpha).round(10))


def rpa_driver(basis, labels, coords, charge, spin):
    na = fermitools.chem.elec.count_alpha(labels, charge, spin)
    nb = fermitools.chem.elec.count_beta(labels, charge, spin)
    nocc = na + nb

    ac, bc = interface.hf.unrestricted_orbitals(basis, labels, coords,
                                                charge, spin)
    nbf = interface.integrals.nbf(basis, labels)
    norb = 2 * nbf
    h_ao = interface.integrals.core_hamiltonian(basis, labels, coords)
    r_ao = interface.integrals.repulsion(basis, labels, coords)

    h_aso = fermitools.math.spinorb.expand(h_ao, brakets=((0, 1),))
    r_aso = fermitools.math.spinorb.expand(r_ao, brakets=((0, 2), (1, 3)))
    g_aso = r_aso - numpy.transpose(r_aso, (0, 1, 3, 2))

    from fermitools.math.spinorb import ab2ov

    c_unsrt = spla.block_diag(ac, bc)
    c = fermitools.math.spinorb.sort(c_unsrt,
                                     order=ab2ov(dim=nbf, na=na, nb=nb),
                                     axes=(1,))
    h = fermitools.math.transform(h_aso, {0: c, 1: c})
    g = fermitools.math.transform(g_aso, {0: c, 1: c, 2: c, 3: c})

    m1 = ocepa0.singles_reference_density(norb=2*nbf, nocc=na+nb)
    m2 = asym("2/3")(numpy.einsum('pr,qs->pqrs', m1, m1))

    a = diagonal_orbital_hessian(nocc=nocc, norb=norb, h=h, g=g, m1=m1, m2=m2)
    b = offdiagonal_orbital_hessian(nocc=nocc, norb=norb, h=h, g=g, m1=m1,
                                    m2=m2)

    w_td = tamm_dancoff_spectrum(a)
    w_rpa = spectrum(a, b)

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


def main():
    CHARGE = 0
    SPIN = 0
    BASIS = 'sto-3g'
    LABELS = ('O', 'H', 'H')
    COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
              (0.000000000000,  1.638036840407,  1.136548822547),
              (0.000000000000, -1.638036840407,  1.136548822547))

    rpa_driver(basis=BASIS, labels=LABELS, coords=COORDS, charge=CHARGE,
               spin=SPIN)

    driver(basis=BASIS, labels=LABELS, coords=COORDS, charge=CHARGE, spin=SPIN)


if __name__ == '__main__':
    main()

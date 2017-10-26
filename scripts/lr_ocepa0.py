import numpy
import scipy.linalg as spla

import fermitools
from fermitools.math.asym import antisymmetrizer_product as asym

import interfaces.psi4 as interface
from . import ocepa0


def second_order_orbital_variation_tensor(h, g, m1, m2):
    i = numpy.eye(*h.shape)
    fc = ocepa0.first_order_orbital_variation_matrix(h, g, m1, m2)
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

    # Solve OCEPA0
    en_nuc = fermitools.chem.nuc.energy(labels=LABELS, coords=COORDS)
    t2_guess = numpy.zeros((nocc, nocc, norb-nocc, norb-nocc))
    en_elec, c, t2 = ocepa0.solve_ocepa0(norb=norb, nocc=nocc, h_aso=h_aso,
                                         g_aso=g_aso, c_guess=c,
                                         t2_guess=t2_guess, niter=200,
                                         e_thresh=1e-14, r_thresh=1e-12,
                                         print_conv=True)
    en_tot = en_elec + en_nuc
    print("Total energy:")
    print('{:20.15f}'.format(en_tot))

    # Build the diagonal orbital and amplitude Hessian
    o = slice(None, nocc)
    v = slice(nocc, None)
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
    en_f = ocepa0.perturbed_energy_function(norb=norb, nocc=nocc, h_aso=h_aso,
                                            p_aso=p_aso, g_aso=g_aso,
                                            c_guess=c, t2_guess=t2, niter=200,
                                            e_thresh=1e-13, r_thresh=1e-9,
                                            print_conv=True)
    en_df2 = fermitools.math.central_difference(en_f, (0., 0., 0.),
                                                step=0.007, nder=2, npts=7)
    print("Compare d2E/df2 to <<mu; mu>>_0:")
    print(en_df2.round(10)/2.)
    print(numpy.diag(alpha).round(10))

    from numpy.testing import assert_almost_equal

    # assert_almost_equal(numpy.diag(alpha), -en_df2, decimal=11)

    # Test the orbital and amplitude gradients
    import functools

    t1 = numpy.zeros((nocc, norb-nocc))
    t2_flat = numpy.ravel(
            fermitools.math.asym.compound_index(t2, {0: (0, 1), 1: (2, 3)}))
    en_func = ocepa0.electronic_energy_functional(norb=norb, nocc=nocc,
                                                  h_aso=h_aso, g_aso=g_aso,
                                                  c=c)

    en_orb = functools.partial(en_func, t2_flat=t2_flat)
    en_amp = functools.partial(en_func, t1)

    en_dorb = numpy.ravel(fermitools.math.central_difference(en_orb, t1,
                                                             step=0.05,
                                                             nder=1,
                                                             npts=9))
    en_damp = numpy.ravel(fermitools.math.central_difference(en_amp, t2_flat,
                                                             step=0.05,
                                                             nder=1,
                                                             npts=9))
    print("Numerical orbital gradient:")
    print(en_dorb.round(9))

    print("Numerical amplitude gradient:")
    print(en_damp.round(9))

    assert_almost_equal(en_dorb, 0., decimal=10)
    assert_almost_equal(en_damp, 0., decimal=10)

    # Test the orbital and amplitude Hessians
    en_dorb2 = numpy.ravel(fermitools.math.central_difference(en_orb, t1,
                                                              step=0.05,
                                                              nder=2,
                                                              npts=9))
    en_damp2 = numpy.ravel(fermitools.math.central_difference(en_amp, t2_flat,
                                                              step=0.05,
                                                              nder=2,
                                                              npts=9))

    print("Numerical orbital Hessian diagonal:")
    print(en_dorb2)

    print("A_orb diagonal:")
    print(numpy.diag(a_orb).round(9))

    print("Ratio:")
    print(numpy.diag(a_orb) / en_dorb2)

    print("Numerical amplitude Hessian diagonal:")
    print(en_damp2)

    print("A_amp diagonal:")
    print(numpy.diag(a_amp).round(9))

    print("Ratio:")
    print(numpy.diag(a_amp) / en_damp2)


if __name__ == '__main__':
    main()

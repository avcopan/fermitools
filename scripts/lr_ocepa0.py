import numpy
import scipy.linalg as spla

import fermitools
from fermitools.math.asym import antisymmetrizer_product as asym

import interfaces.psi4 as interface
from .ocepa0 import first_order_orbital_variation_matrix


def second_order_orbital_variation_tensor(h, g, m1, m2):
    i = numpy.eye(*h.shape)
    fc = first_order_orbital_variation_matrix(h, g, m1, m2)
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


def mixed_interaction(fov, gooov, govvv):
    no, nv, _, _ = govvv.shape
    io = numpy.eye(no)
    iv = numpy.eye(nv)
    ioo = (+ numpy.einsum('ik,la->iakl', io, fov)
           - numpy.einsum('ilka->iakl', gooov))
    ivv = (- numpy.einsum('ac,id->iadc', iv, fov)
           + numpy.einsum('icad->iadc', govvv))
    return {'o,o': ioo, 'v,v': ivv}


def diagonal_mixed_hessian(fov, gooov, govvv, t2):
    no, _, nv, _ = t2.shape
    nsingles = no * nv
    ndoubles = no * (no - 1) * nv * (nv - 1) // 4
    io = numpy.eye(no)
    iv = numpy.eye(nv)
    i = mixed_interaction(fov, gooov, govvv)
    a = (- asym('2/3')(
                numpy.einsum('ik,lacd->iaklcd', io, govvv))
         - asym('4/5')(
                numpy.einsum('ac,klid->iaklcd', iv, gooov))
         - asym('2/3')(
                numpy.einsum('iakm,mlcd->iaklcd', i['o,o'], t2))
         + asym('4/5')(
                numpy.einsum('iaec,kled->iaklcd', i['v,v'], t2))
         - asym('2/3|4/5')(
                numpy.einsum('ik,mcae,mled->iaklcd', io, govvv, t2))
         - asym('2/3|4/5')(
                numpy.einsum('ac,imke,mled->iaklcd', iv, gooov, t2))
         - 1./2 * asym('2/3')(
                numpy.einsum('ik,mnla,mncd->iaklcd', io, gooov, t2))
         - 1./2 * asym('4/5')(
                numpy.einsum('ac,idef,klef->iaklcd', iv, govvv, t2)))
    a_cmp = fermitools.math.asym.compound_index(a, {2: (2, 3), 3: (4, 5)})
    return numpy.reshape(a_cmp, (nsingles, ndoubles))


def offdiagonal_mixed_hessian(fov, gooov, govvv, t2):
    no, _, nv, _ = t2.shape
    nsingles = no * nv
    ndoubles = no * (no - 1) * nv * (nv - 1) // 4
    i = mixed_interaction(fov, gooov, govvv)
    b = (- asym('2/3')(
                numpy.einsum('iamk,mlcd->iaklcd', i['o,o'], t2))
         + asym('4/5')(
                numpy.einsum('iace,kled->iaklcd', i['v,v'], t2))
         - asym('2/3|4/5')(
                numpy.einsum('lead,kice->iaklcd', govvv, t2))
         - asym('2/3|4/5')(
                numpy.einsum('ilmd,kmca->iaklcd', gooov, t2))
         + numpy.einsum('klma,micd->iaklcd', gooov, t2)
         + numpy.einsum('iecd,klea->iaklcd', govvv, t2))
    b_cmp = fermitools.math.asym.compound_index(b, {2: (2, 3), 3: (4, 5)})
    return numpy.reshape(b_cmp, (nsingles, ndoubles))


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
    r, _, _, _ = spla.lstsq(a + b, 2 * t)
    return r


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


def main():
    import scripts.ocepa0 as ocepa0

    import numpy
    import scipy.linalg as spla

    import fermitools

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

    # Solve OCEPA0
    t2_guess = numpy.zeros((nocc, nocc, norb-nocc, norb-nocc))
    en_elec, c, t2 = ocepa0.solve(norb=norb, nocc=nocc, h_aso=h_aso,
                                  g_aso=g_aso, c_guess=c,
                                  t2_guess=t2_guess, niter=200,
                                  e_thresh=1e-14, r_thresh=1e-13,
                                  print_conv=True)

    # Build the diagonal orbital and amplitude Hessian
    o = slice(None, nocc)
    v = slice(nocc, None)
    h = fermitools.math.transform(h_aso, {0: c, 1: c})
    g = fermitools.math.transform(g_aso, {0: c, 1: c, 2: c, 3: c})
    m1_ref = ocepa0.singles_reference_density(norb=norb, nocc=nocc)
    f = ocepa0.fock(h, g, m1_ref)
    m1_cor = ocepa0.singles_correlation_density(t2)
    m1 = m1_ref + m1_cor
    k2 = ocepa0.doubles_cumulant(t2)
    m2 = ocepa0.doubles_density(m1_ref, m1_cor, k2)

    a_orb = diagonal_orbital_hessian(nocc, norb, h, g, m1, m2)
    a_mix = diagonal_mixed_hessian(f[o, v], g[o, o, o, v], g[o, v, v, v], t2)
    a_amp = diagonal_amplitude_hessian(f[o, o], f[v, v], g[o, o, o, o],
                                       g[o, v, o, v], g[v, v, v, v])

    b_orb = offdiagonal_orbital_hessian(nocc, norb, h, g, m1, m2)
    b_mix = offdiagonal_mixed_hessian(f[o, v], g[o, o, o, v], g[o, v, v, v],
                                      t2)
    b_amp = numpy.zeros_like(a_amp)

    # Evaluate dipole polarizability using linear response theory
    p_ao = interface.integrals.dipole(BASIS, LABELS, COORDS)
    p_aso = fermitools.math.spinorb.expand(p_ao, brakets=((1, 2),))
    p = fermitools.math.transform(p_aso, {1: c, 2: c})
    t_orb = numpy.transpose([
        orbital_property_gradient(o, v, px, m1) for px in p])
    t_amp = numpy.transpose([
        amplitude_property_gradient(px[o, o], px[v, v], t2) for px in p])

    a = numpy.bmat([[a_orb, -a_mix], [-a_mix.T, a_amp]])
    b = numpy.bmat([[b_orb, -b_mix], [-b_mix.T, b_amp]])
    t = numpy.bmat([[t_orb], [t_amp]])
    r = static_response_vector(a, b, t)
    alpha = static_linear_response_function(t, r)

    print(numpy.real(alpha).round(8))

    # Evaluate dipole polarizability as energy derivative
    en_f_func = ocepa0.perturbed_energy_function(norb=norb, nocc=nocc,
                                                 h_aso=h_aso, p_aso=p_aso,
                                                 g_aso=g_aso, c_guess=c,
                                                 t2_guess=t2, niter=200,
                                                 e_thresh=1e-14,
                                                 r_thresh=1e-12,
                                                 print_conv=True)
    en_df2 = fermitools.math.central_difference(en_f_func, [0., 0., 0.],
                                                step=0.01, nder=2, npts=9)

    print(numpy.diag(numpy.real(alpha)).round(8))
    print(en_df2.round(8))


if __name__ == '__main__':
    main()

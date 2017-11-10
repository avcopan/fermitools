import numpy
import scipy.linalg as spla

import fermitools
from fermitools.math import einsum

import interfaces.psi4 as interface
from . import scf


def diagonal_orbital_hessian(hoo, hvv, goooo, goovv, govov, gvvvv, m1oo, m1vv,
                             m2oooo, m2oovv, m2ovov, m2vvvv):
    no, nv, _, _ = govov.shape
    nsingles = no * nv
    io = numpy.eye(no)
    iv = numpy.eye(nv)
    fcoo = (numpy.dot(hoo, m1oo)
            + 1./2 * einsum('imno,jmno->ij', goooo, m2oooo)
            + 1./2 * einsum('imef,jmef->ij', goovv, m2oovv)
            + einsum('iemf,jemf->ij', govov, m2ovov))
    fcvv = (numpy.dot(hvv, m1vv)
            + einsum('nema,nemb->ab', govov, m2ovov)
            + 1./2 * einsum('mnae,mnbe', goovv, m2oovv)
            + 1./2 * einsum('aefg,befg', gvvvv, m2vvvv))
    fsoo = (fcoo + numpy.transpose(fcoo)) / 2.
    fsvv = (fcvv + numpy.transpose(fcvv)) / 2.
    a = (+ einsum('ij,ab->iajb', hoo, m1vv)
         + einsum('ij,ab->iajb', m1oo, hvv)
         - einsum('ij,ab->iajb', io, fsvv)
         - einsum('ij,ab->iajb', fsoo, iv)
         + einsum('minj,manb->iajb', goooo, m2ovov)
         + einsum('minj,manb->iajb', m2oooo, govov)
         + einsum('iejf,aebf->iajb', govov, m2vvvv)
         + einsum('iejf,aebf->iajb', m2ovov, gvvvv)
         + einsum('ibme,jame->iajb', govov, m2ovov)
         + einsum('ibme,jame->iajb', m2ovov, govov))
    return numpy.reshape(a, (nsingles, nsingles))
# def diagonal_orbital_hessian_operator(hoo, hvv, goooo, goovv, govov, gvvvv,
#                                       m1oo, m1vv, m2oooo, m2oovv, m2ovov,
#                                       m2vvvv):
#     fcoo = (numpy.dot(hoo, m1oo)
#             + 1./2 * einsum('imno,jmno->ij', goooo, m2oooo)
#             + 1./2 * einsum('imef,jmef->ij', goovv, m2oovv)
#             + einsum('iemf,jemf->ij', govov, m2ovov))
#     fcvv = (numpy.dot(hvv, m1vv)
#             + einsum('nema,nemb->ab', govov, m2ovov)
#             + 1./2 * einsum('mnae,mnbe', goovv, m2oovv)
#             + 1./2 * einsum('aefg,befg', gvvvv, m2vvvv))
#     fsoo = (fcoo + numpy.transpose(fcoo)) / 2.
#     fsvv = (fcvv + numpy.transpose(fcvv)) / 2.
#     def _sigma(r1):


def offdiagonal_orbital_hessian(goooo, goovv, govov, gvvvv, m2oooo, m2oovv,
                                m2ovov, m2vvvv):
    no, nv, _, _ = govov.shape
    nsingles = no * nv
    b = (+ einsum('imbe,jema->iajb', goovv, m2ovov)
         + einsum('imbe,jema->iajb', m2oovv, govov)
         + einsum('iemb,jmae->iajb', govov, m2oovv)
         + einsum('iemb,jmae->iajb', m2ovov, goovv)
         + 1./2 * einsum('ijmn,mnab->iajb', goooo, m2oovv)
         + 1./2 * einsum('ijmn,mnab->iajb', m2oooo, goovv)
         + 1./2 * einsum('ijef,efab->iajb', goovv, m2vvvv)
         + 1./2 * einsum('ijef,efab->iajb', m2oovv, gvvvv))
    return numpy.reshape(b, (nsingles, nsingles))


def orbital_property_gradient(pov, m1oo, m1vv):
    no, _ = m1oo.shape
    nv, _ = m1vv.shape
    nsingles = no * nv
    t = (+ einsum('...ie,ea->ia...', pov, m1vv)
         - einsum('im,...ma->ia...', m1oo, pov))
    shape = (nsingles,) + t.shape[2:]
    return numpy.reshape(t, shape)


def orbital_metric(m1oo, m1vv):
    no, _ = m1oo.shape
    nv, _ = m1vv.shape
    nsingles = no * nv
    io = numpy.eye(no)
    iv = numpy.eye(nv)
    s = (+ einsum('ij,ab->iajb', m1oo, iv)
         - einsum('ij,ab->iajb', io, m1vv))
    return numpy.reshape(s, (nsingles, nsingles))


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
    return numpy.tensordot(t, r, axes=(0, 0))


def spectrum(a, b):
    w2 = spla.eigvals(numpy.dot(a + b, a - b))
    return numpy.array(sorted(numpy.sqrt(w2.real)))


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
    en_elec, c = scf.solve(norb=norb, nocc=nocc, h_aso=h_aso, g_aso=g_aso,
                           c_guess=c, niter=200, e_thresh=1e-14,
                           r_thresh=1e-12, print_conv=200)
    en_tot = en_elec + en_nuc
    print("Total energy:")
    print('{:20.15f}'.format(en_tot))

    # Evalute the dipole polarizability as a linear response function
    o = slice(None, nocc)
    v = slice(nocc, None)
    h = fermitools.math.transform(h_aso, {0: c, 1: c})
    g = fermitools.math.transform(g_aso, {0: c, 1: c, 2: c, 3: c})
    m1 = scf.singles_density(norb=norb, nocc=nocc)
    m2 = scf.doubles_density(m1)
    a_orb = diagonal_orbital_hessian(h[o, o], h[v, v], g[o, o, o, o],
                                     g[o, o, v, v], g[o, v, o, v],
                                     g[v, v, v, v], m1[o, o], m1[v, v],
                                     m2[o, o, o, o], m2[o, o, v, v],
                                     m2[o, v, o, v], m2[v, v, v, v])
    b_orb = offdiagonal_orbital_hessian(g[o, o, o, o], g[o, o, v, v],
                                        g[o, v, o, v], g[v, v, v, v],
                                        m2[o, o, o, o], m2[o, o, v, v],
                                        m2[o, v, o, v], m2[v, v, v, v])

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
    from numpy.testing import assert_almost_equal

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'data')
    en_dxdx = numpy.load(os.path.join(data_path,
                                      'lr_scf/en_dxdx.npy'))

    print("Checking orbital Hessian:")
    print(numpy.diag(a_orb + b_orb) / numpy.diag(en_dxdx))
    print(spla.norm(en_dxdx - 2*(a_orb + b_orb)))

    assert_almost_equal(en_dxdx, 2*(a_orb + b_orb), decimal=9)

    # Evaluate dipole polarizability using linear response theory
    p_ao = interface.integrals.dipole(BASIS, LABELS, COORDS)
    p_aso = fermitools.math.spinorb.expand(p_ao, brakets=((1, 2),))
    p = fermitools.math.transform(p_aso, {1: c, 2: c})
    t_orb = orbital_property_gradient(p[:, o, v], m1[o, o], m1[v, v])
    r_orb = static_response_vector(a_orb, b_orb, t_orb)
    alpha = static_linear_response_function(t_orb, r_orb)

    print(numpy.real(alpha).round(8))

    # Evaluate dipole polarizability as energy derivative
    en_f_func = scf.perturbed_energy_function(norb=norb, nocc=nocc,
                                              h_aso=h_aso, p_aso=p_aso,
                                              g_aso=g_aso, c_guess=c,
                                              niter=200, e_thresh=1e-14,
                                              r_thresh=1e-12, print_conv=True)
    en_df2 = fermitools.math.central_difference(en_f_func, [0., 0., 0.],
                                                step=0.01, nder=2, npts=9)
    print("Compare d2E/df2 to <<mu; mu>>_0:")
    print(en_df2.round(10))
    print(numpy.diag(alpha).round(10))
    print((numpy.diag(alpha) / en_df2))

    assert_almost_equal(numpy.diag(alpha), -en_df2, decimal=9)

    # Evaluate the excitation energies
    s_orb = orbital_metric(m1[o, o], m1[v, v])

    e_orb = numpy.bmat([[a_orb, b_orb], [b_orb, a_orb]])
    m_orb = spla.block_diag(s_orb, -s_orb)
    w_orb = numpy.array(sorted(numpy.real(spla.eigvals(e_orb, b=m_orb))))
    pos_idx = numpy.nonzero(w_orb > 0.)
    w_orb_pos = w_orb[pos_idx]
    print(w_orb_pos)
    assert_almost_equal(w_orb_pos, w_rpa_ref, decimal=10)


if __name__ == '__main__':
    main()

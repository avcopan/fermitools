import numpy
import scipy.linalg
import scipy.sparse.linalg

import warnings

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


def diagonal_orbital_metric(m1oo, m1vv):
    no, _ = m1oo.shape
    nv, _ = m1vv.shape
    nsingles = no * nv
    io = numpy.eye(no)
    iv = numpy.eye(nv)
    s = (+ einsum('ij,ab->iajb', m1oo, iv)
         - einsum('ij,ab->iajb', io, m1vv))
    return numpy.reshape(s, (nsingles, nsingles))


def diagonal_orbital_hessian_sigma(hoo, hvv, goooo, goovv, govov, gvvvv,
                                   m1oo, m1vv, m2oooo, m2oovv, m2ovov,
                                   m2vvvv):
    no, nv, _, _ = govov.shape
    nsingles = no * nv
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

    def _sigma(r1_flat):
        cols = 1 if r1_flat.ndim is 1 else r1_flat.shape[1]
        r1 = numpy.reshape(r1_flat, (no, nv, cols))
        a_times_r1 = (+ einsum('ij,ab,jbx->iax', hoo, m1vv, r1)
                      + einsum('ij,ab,jbx->iax', m1oo, hvv, r1)
                      - einsum('ab,ibx->iax', fsvv, r1)
                      - einsum('ij,jax->iax', fsoo, r1)
                      + einsum('minj,manb,jbx->iax', goooo, m2ovov, r1)
                      + einsum('minj,manb,jbx->iax', m2oooo, govov, r1)
                      + einsum('iejf,aebf,jbx->iax', govov, m2vvvv, r1)
                      + einsum('iejf,aebf,jbx->iax', m2ovov, gvvvv, r1)
                      + einsum('ibme,jame,jbx->iax', govov, m2ovov, r1)
                      + einsum('ibme,jame,jbx->iax', m2ovov, govov, r1))
        return numpy.squeeze(numpy.reshape(a_times_r1, (nsingles, cols)))

    return _sigma


def offdiagonal_orbital_hessian_sigma(goooo, goovv, govov, gvvvv, m2oooo,
                                      m2oovv, m2ovov, m2vvvv):
    no, nv, _, _ = govov.shape
    nsingles = no * nv

    def _sigma(r1_flat):
        cols = 1 if r1_flat.ndim is 1 else r1_flat.shape[1]
        r1 = numpy.reshape(r1_flat, (no, nv, cols))
        b_times_r1 = (+ einsum('imbe,jema,jbx->iax', goovv, m2ovov, r1)
                      + einsum('imbe,jema,jbx->iax', m2oovv, govov, r1)
                      + einsum('iemb,jmae,jbx->iax', govov, m2oovv, r1)
                      + einsum('iemb,jmae,jbx->iax', m2ovov, goovv, r1)
                      + 1./2 * einsum('ijmn,mnab,jbx->iax', goooo, m2oovv, r1)
                      + 1./2 * einsum('ijmn,mnab,jbx->iax', m2oooo, goovv, r1)
                      + 1./2 * einsum('ijef,efab,jbx->iax', goovv, m2vvvv, r1)
                      + 1./2 * einsum('ijef,efab,jbx->iax', m2oovv, gvvvv, r1))
        return numpy.squeeze(numpy.reshape(b_times_r1, (nsingles, cols)))

    return _sigma


def diagonal_orbital_metric_sigma(m1oo, m1vv):
    no, _ = m1oo.shape
    nv, _ = m1vv.shape
    nsingles = no * nv

    def _sigma(r1_flat):
        cols = 1 if r1_flat.ndim is 1 else r1_flat.shape[1]
        r1 = numpy.reshape(r1_flat, (no, nv, cols))
        s_times_r1 = (+ einsum('ij,jax->iax', m1oo, r1)
                      - einsum('ab,ibx->iax', m1vv, r1))
        return numpy.squeeze(numpy.reshape(s_times_r1, (nsingles, cols)))

    return _sigma


def orbital_property_gradient(pov, m1oo, m1vv):
    no, _ = m1oo.shape
    nv, _ = m1vv.shape
    nsingles = no * nv
    t = (+ einsum('...ie,ea->ia...', pov, m1vv)
         - einsum('im,...ma->ia...', m1oo, pov))
    shape = (nsingles,) + t.shape[2:]
    return numpy.reshape(t, shape)


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
    r, _, _, _ = scipy.linalg.lstsq(a + b, 2 * t)
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


def solve_static_response_vector(no, nv, sig_a, sig_b, t):
    nsingles = no * nv
    a_ = scipy.sparse.linalg.LinearOperator((nsingles, nsingles), matvec=sig_a)
    b_ = scipy.sparse.linalg.LinearOperator((nsingles, nsingles), matvec=sig_b)

    def _solve(t_):
        r, info = scipy.sparse.linalg.cg(a_ + b_, 2 * t_)
        if info != 0:
            warnings.warn("Did not converge!  Output code {:d}".format(info))
        return r

    ts = map(numpy.moveaxis(t, 0, -1).__getitem__, numpy.ndindex(t.shape[1:]))
    rs = tuple(map(_solve, ts))
    return numpy.reshape(numpy.moveaxis(rs, -1, 0), t.shape)


def solve_spectrum(no, nv, sig_a, sig_b, sig_s, k=6):
    nsingles = no * nv

    def _sig_e(x1y1):
        x1, y1 = x1y1[:nsingles], x1y1[nsingles:]
        return numpy.concatenate((+sig_a(x1) + sig_b(y1),
                                  -sig_b(x1) - sig_a(y1)), axis=0)

    def _sig_m(x1y1):
        x1, y1 = x1y1[:nsingles], x1y1[nsingles:]
        return numpy.concatenate((sig_s(x1), sig_s(y1)), axis=0)

    e_ = scipy.sparse.linalg.LinearOperator((2*nsingles, 2*nsingles),
                                            matvec=_sig_e)
    m_ = scipy.sparse.linalg.LinearOperator((2*nsingles, 2*nsingles),
                                            matvec=_sig_m)

    return scipy.sparse.linalg.eigs(e_, k=k, M=m_, which='SM')


def spectrum(a, b):
    w2 = scipy.linalg.eigvals(numpy.dot(a + b, a - b))
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
    c_unsrt = scipy.linalg.block_diag(ac, bc)
    sortvec = fermitools.math.spinorb.ab2ov(dim=nbf, na=na, nb=nb)
    c_unsrt = scipy.linalg.block_diag(ac, bc)
    c = fermitools.math.spinorb.sort(c_unsrt, order=sortvec, axes=(1,))

    en_nuc = fermitools.chem.nuc.energy(labels=LABELS, coords=COORDS)
    en_elec, c = scf.solve(norb=norb, nocc=nocc, h_aso=h_aso, g_aso=g_aso,
                           c_guess=c, niter=200, e_thresh=1e-14,
                           r_thresh=1e-12, print_conv=200)
    en_tot = en_elec + en_nuc
    print("Total energy:")
    print('{:20.15f}'.format(en_tot))

    # Evalute the dipole polarizability as a linear response function
    no = nocc
    nv = norb - nocc
    nsingles = no * nv
    o = slice(None, nocc)
    v = slice(nocc, None)
    h = fermitools.math.transform(h_aso, {0: c, 1: c})
    g = fermitools.math.transform(g_aso, {0: c, 1: c, 2: c, 3: c})
    m1 = scf.singles_density(norb=norb, nocc=nocc)
    m2 = scf.doubles_density(m1)
    a = diagonal_orbital_hessian(h[o, o], h[v, v], g[o, o, o, o],
                                 g[o, o, v, v], g[o, v, o, v],
                                 g[v, v, v, v], m1[o, o], m1[v, v],
                                 m2[o, o, o, o], m2[o, o, v, v],
                                 m2[o, v, o, v], m2[v, v, v, v])
    b = offdiagonal_orbital_hessian(g[o, o, o, o], g[o, o, v, v],
                                    g[o, v, o, v], g[v, v, v, v],
                                    m2[o, o, o, o], m2[o, o, v, v],
                                    m2[o, v, o, v], m2[v, v, v, v])
    s = diagonal_orbital_metric(m1[o, o], m1[v, v])

    e = numpy.bmat([[a, b], [-b, -a]])
    m = scipy.linalg.block_diag(s, s)
    w_old, u_old = scipy.linalg.eig(e, b=m)
    w_old = numpy.real(sorted(w_old))

    # Evaluate dipole polarizability using linear response theory
    p_ao = interface.integrals.dipole(BASIS, LABELS, COORDS)
    p_aso = fermitools.math.spinorb.expand(p_ao, brakets=((1, 2),))
    p = fermitools.math.transform(p_aso, {1: c, 2: c})
    t = orbital_property_gradient(p[:, o, v], m1[o, o], m1[v, v])
    r = static_response_vector(a, b, t)
    alpha_old = static_linear_response_function(t, r)
    sig_a = diagonal_orbital_hessian_sigma(h[o, o], h[v, v], g[o, o, o, o],
                                           g[o, o, v, v], g[o, v, o, v],
                                           g[v, v, v, v], m1[o, o], m1[v, v],
                                           m2[o, o, o, o], m2[o, o, v, v],
                                           m2[o, v, o, v], m2[v, v, v, v])
    sig_b = offdiagonal_orbital_hessian_sigma(g[o, o, o, o], g[o, o, v, v],
                                              g[o, v, o, v], g[v, v, v, v],
                                              m2[o, o, o, o], m2[o, o, v, v],
                                              m2[o, v, o, v], m2[v, v, v, v])
    sig_s = diagonal_orbital_metric_sigma(m1[o, o], m1[v, v])

    i = numpy.eye(nsingles)
    print("a:")
    print(numpy.linalg.norm(a - sig_a(i)))
    print("b:")
    print(numpy.linalg.norm(b - sig_b(i)))
    print("s:")
    print(numpy.linalg.norm(s - sig_s(i)))

    from numpy.testing import assert_almost_equal

    # Excitation energies
    w, u = solve_spectrum(no, nv, sig_a, sig_b, sig_s, k=2*nsingles-2)
    w = numpy.real(sorted(w))
    print(w / w_old[1:-1])
    assert_almost_equal(w, w_old[1:-1], decimal=12)

    # Response function
    r = solve_static_response_vector(no, nv, sig_a, sig_b, t)
    alpha = numpy.tensordot(r, t, axes=(0, 0))
    print(numpy.diag(alpha) / numpy.diag(alpha_old))
    assert_almost_equal(alpha, alpha_old, decimal=12)


if __name__ == '__main__':
    main()

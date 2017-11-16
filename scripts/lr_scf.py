import numpy
import scipy.linalg
import scipy.sparse.linalg

import warnings

import fermitools
from fermitools.math import einsum

import interfaces.psi4 as interface
from . import scf


def orbital_hessian_diag(hoo, hvv, goooo, goovv, govov, gvvvv, m1oo, m1vv,
                         m2oooo, m2oovv, m2ovov, m2vvvv):
    io = numpy.eye(*hoo.shape)
    iv = numpy.eye(*hvv.shape)
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
    return (
        + einsum('ij,ab->iajb', hoo, m1vv)
        + einsum('ij,ab->iajb', m1oo, hvv)
        - einsum('ij,ab->iajb', io, fsvv)
        - einsum('ij,ab->iajb', fsoo, iv)
        + einsum('minj,manb->iajb', goooo, m2ovov)
        + einsum('minj,manb->iajb', m2oooo, govov)
        + einsum('iejf,aebf->iajb', govov, m2vvvv)
        + einsum('iejf,aebf->iajb', m2ovov, gvvvv)
        + einsum('ibme,jame->iajb', govov, m2ovov)
        + einsum('ibme,jame->iajb', m2ovov, govov))


def orbital_hessian_offd(goooo, goovv, govov, gvvvv, m2oooo, m2oovv, m2ovov,
                         m2vvvv):
    return (
        + einsum('imbe,jema->iajb', goovv, m2ovov)
        + einsum('imbe,jema->iajb', m2oovv, govov)
        + einsum('iemb,jmae->iajb', govov, m2oovv)
        + einsum('iemb,jmae->iajb', m2ovov, goovv)
        + 1./2 * einsum('ijmn,mnab->iajb', goooo, m2oovv)
        + 1./2 * einsum('ijmn,mnab->iajb', m2oooo, goovv)
        + 1./2 * einsum('ijef,efab->iajb', goovv, m2vvvv)
        + 1./2 * einsum('ijef,efab->iajb', m2oovv, gvvvv))


def orbital_metric(m1oo, m1vv):
    io = numpy.eye(*m1oo.shape)
    iv = numpy.eye(*m1vv.shape)
    return (
        + einsum('ij,ab->iajb', m1oo, iv)
        - einsum('ij,ab->iajb', io, m1vv))


def orbital_property_gradient(pov, m1oo, m1vv):
    return (
        + einsum('...ie,ea->ia...', pov, m1vv)
        - einsum('im,...ma->ia...', m1oo, pov))


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
    r, _, _, _ = scipy.linalg.lstsq(a + b, -2 * t)
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
    w2 = scipy.linalg.eigvals(numpy.dot(a + b, a - b))
    return numpy.array(sorted(numpy.sqrt(w2.real)))


def orbital_hessian_diag_sigma(hoo, hvv, goooo, goovv, govov, gvvvv, m1oo,
                               m1vv, m2oooo, m2oovv, m2ovov, m2vvvv):
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

    def _sigma(r1):
        return (
            # a terms
            + einsum('ij,ab,jb...->ia...', hoo, m1vv, r1)
            + einsum('ij,ab,jb...->ia...', m1oo, hvv, r1)
            - einsum('ab,ib...->ia...', fsvv, r1)
            - einsum('ij,ja...->ia...', fsoo, r1)
            + einsum('minj,manb,jb...->ia...', goooo, m2ovov, r1)
            + einsum('minj,manb,jb...->ia...', m2oooo, govov, r1)
            + einsum('iejf,aebf,jb...->ia...', govov, m2vvvv, r1)
            + einsum('iejf,aebf,jb...->ia...', m2ovov, gvvvv, r1)
            + einsum('ibme,jame,jb...->ia...', govov, m2ovov, r1)
            + einsum('ibme,jame,jb...->ia...', m2ovov, govov, r1))

    return _sigma


def orbital_hessian_offd_sigma(goooo, goovv, govov, gvvvv, m2oooo, m2oovv,
                               m2ovov, m2vvvv):

    def _sigma(r1):
        return (
            + einsum('imbe,jema,jb...->ia...', goovv, m2ovov, r1)
            + einsum('imbe,jema,jb...->ia...', m2oovv, govov, r1)
            + einsum('iemb,jmae,jb...->ia...', govov, m2oovv, r1)
            + einsum('iemb,jmae,jb...->ia...', m2ovov, goovv, r1)
            + 1./2 * einsum('ijmn,mnab,jb...->ia...', goooo, m2oovv, r1)
            + 1./2 * einsum('ijmn,mnab,jb...->ia...', m2oooo, goovv, r1)
            + 1./2 * einsum('ijef,efab,jb...->ia...', goovv, m2vvvv, r1)
            + 1./2 * einsum('ijef,efab,jb...->ia...', m2oovv, gvvvv, r1))

    return _sigma


def effective_response_hamiltonian_sigma(sig_e_sum, sig_e_diff, sig_s_inv):

    def _sigma(z):
        return sig_s_inv(sig_e_sum(sig_s_inv(sig_e_diff(z))))

    return _sigma


def solve_static_response_vector(dim, sig_e_sum, t):
    e_ = scipy.sparse.linalg.LinearOperator((dim, dim), matvec=sig_e_sum)

    def _solve(t_):
        r, info = scipy.sparse.linalg.cg(e_, -2 * t_)
        if info != 0:
            warnings.warn("Did not converge!  Output code {:d}".format(info))
        return r

    ts = map(numpy.moveaxis(t, 0, -1).__getitem__, numpy.ndindex(t.shape[1:]))
    rs = tuple(map(_solve, ts))
    return numpy.reshape(numpy.moveaxis(rs, -1, 0), t.shape)


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
    ac, bc = interface.hf.unrestricted_orbitals(
            BASIS, LABELS, COORDS, CHARGE, SPIN)
    c_unsrt = scipy.linalg.block_diag(ac, bc)
    sortvec = fermitools.math.spinorb.ab2ov(dim=nbf, na=na, nb=nb)
    c_unsrt = scipy.linalg.block_diag(ac, bc)
    c = fermitools.math.spinorb.sort(c_unsrt, order=sortvec, axes=(1,))

    en_nuc = fermitools.chem.nuc.energy(labels=LABELS, coords=COORDS)
    en_elec, c = scf.solve(
            norb=norb, nocc=nocc, h_aso=h_aso, g_aso=g_aso, c_guess=c,
            niter=200, e_thresh=1e-14, r_thresh=1e-12, print_conv=200)
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

    v_raveler = fermitools.math.raveler({0: (0, 1)})
    m_raveler = fermitools.math.raveler({0: (0, 1), 1: (2, 3)})

    def v_unraveler(r1):
        shape = (no, nv) if r1.ndim == 1 else (no, nv) + r1.shape[1:]
        return numpy.reshape(r1, shape)

    def add(f, g):

        def _sum(*args, **kwargs):
            return f(*args, **kwargs) + g(*args, **kwargs)

        return _sum

    def sub(f, g):

        def _diff(*args, **kwargs):
            return f(*args, **kwargs) - g(*args, **kwargs)

        return _diff

    a = m_raveler(orbital_hessian_diag(
            h[o, o], h[v, v], g[o, o, o, o], g[o, o, v, v], g[o, v, o, v],
            g[v, v, v, v], m1[o, o], m1[v, v], m2[o, o, o, o], m2[o, o, v, v],
            m2[o, v, o, v], m2[v, v, v, v]))
    b = m_raveler(orbital_hessian_offd(
            g[o, o, o, o], g[o, o, v, v], g[o, v, o, v], g[v, v, v, v],
            m2[o, o, o, o], m2[o, o, v, v], m2[o, v, o, v], m2[v, v, v, v]))
    s = m_raveler(orbital_metric(m1[o, o], m1[v, v]))

    e = numpy.bmat([[a, b], [-b, -a]])
    m = scipy.linalg.block_diag(s, s)
    w_old, u_old = scipy.linalg.eig(e, b=m)
    w_old = numpy.real(sorted(w_old))

    # Evaluate dipole polarizability using linear response theory
    p_ao = interface.integrals.dipole(BASIS, LABELS, COORDS)
    p_aso = fermitools.math.spinorb.expand(p_ao, brakets=((1, 2),))
    p = fermitools.math.transform(p_aso, {1: c, 2: c})
    t = v_raveler(orbital_property_gradient(p[:, o, v], m1[o, o], m1[v, v]))
    r = static_response_vector(a, b, t)
    alpha_old = static_linear_response_function(t, r)

    from numpy.testing import assert_almost_equal
    from toolz import functoolz

    # Solve positive excitation energies only
    sig_a = orbital_hessian_diag_sigma(
            h[o, o], h[v, v], g[o, o, o, o], g[o, o, v, v], g[o, v, o, v],
            g[v, v, v, v], m1[o, o], m1[v, v], m2[o, o, o, o], m2[o, o, v, v],
            m2[o, v, o, v], m2[v, v, v, v])
    sig_b = orbital_hessian_offd_sigma(
            g[o, o, o, o], g[o, o, v, v], g[o, v, o, v], g[v, v, v, v],
            m2[o, o, o, o], m2[o, o, v, v], m2[o, v, o, v], m2[v, v, v, v])
    sig_e_sum = functoolz.compose(
            v_raveler,
            add(sig_a, sig_b),
            v_unraveler)
    sig_e_diff = functoolz.compose(
            v_raveler,
            sub(sig_a, sig_b),
            v_unraveler)

    i1 = numpy.eye(nsingles)
    assert_almost_equal(sig_e_sum(i1), a + b)
    assert_almost_equal(sig_e_diff(i1), a - b)

    sig_s_inv = scipy.sparse.linalg.aslinearoperator(scipy.linalg.inv(s))
    sig_e_eff = effective_response_hamiltonian_sigma(
            sig_e_sum, sig_e_diff, sig_s_inv)

    e_ = scipy.sparse.linalg.LinearOperator(
            (nsingles, nsingles), matvec=sig_e_eff)
    w2, u = scipy.sparse.linalg.eigs(e_, k=nsingles-2, which='SR')
    w = numpy.sqrt(numpy.real(sorted(w2)))
    print(w / w_old[nsingles:-2])

    # Response function
    r = solve_static_response_vector(nsingles, sig_e_sum, t)
    alpha = numpy.tensordot(r, t, axes=(0, 0))
    print(numpy.diag(alpha) / numpy.diag(alpha_old))
    assert_almost_equal(alpha, alpha_old, decimal=12)


if __name__ == '__main__':
    main()

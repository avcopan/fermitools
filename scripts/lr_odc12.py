import numpy
import scipy.linalg
from toolz import functoolz

import fermitools
from fermitools.math import einsum
from fermitools.math.asym import antisymmetrizer_product as asym

import interfaces.psi4 as interface
from .odc12 import doubles_cumulant
from .odc12 import doubles_density
from .lr_ocepa0 import orbital_hessian_diag
from .lr_ocepa0 import orbital_hessian_offd
from .lr_ocepa0 import amplitude_hessian as cepa_amplitude_hessian
from .lr_ocepa0 import orbital_property_gradient
from .lr_ocepa0 import amplitude_property_gradient
from .lr_ocepa0 import static_response_vector
from .lr_ocepa0 import static_linear_response_function
from .lr_ocepa0 import orbital_metric
from .lr_ocepa0 import (amplitude_hessian_sigma
                        as cepa_amplitude_hessian_sigma)
from .lr_ocepa0 import orbital_hessian_diag_sigma
from .lr_ocepa0 import orbital_hessian_offd_sigma
from .lr_ocepa0 import combined_hessian_sigma
from .lr_ocepa0 import combined_metric_sigma
from .lr_ocepa0 import effective_response_hamiltonian_sigma
from .lr_ocepa0 import solve_static_response_vector


def fancy_repulsion(ffoo, ffvv, goooo, govov, gvvvv, m1oo, m1vv):
    no, uo = scipy.linalg.eigh(m1oo)
    nv, uv = scipy.linalg.eigh(m1vv)
    n1oo = fermitools.math.broadcast_sum({0: no, 1: no}) - 1
    n1vv = fermitools.math.broadcast_sum({0: nv, 1: nv}) - 1
    io = numpy.eye(*uo.shape)
    iv = numpy.eye(*uv.shape)
    tffoo = fermitools.math.transform(ffoo, {0: uo, 1: uo})
    tffvv = fermitools.math.transform(ffvv, {0: uv, 1: uv})
    tgoooo = fermitools.math.transform(goooo, {0: uo, 1: uo, 2: uo, 3: uo})
    tgovov = fermitools.math.transform(govov, {0: uo, 1: uv, 2: uo, 3: uv})
    tgvvvv = fermitools.math.transform(gvvvv, {0: uv, 1: uv, 2: uv, 3: uv})
    tfgoooo = ((tgoooo - einsum('il,jk->ikjl', tffoo, io)
                       - einsum('il,jk->ikjl', io, tffoo))
               / einsum('ij,kl->ikjl', n1oo, n1oo))
    tfgovov = tgovov / einsum('ij,ab->iajb', n1oo, n1vv)
    tfgvvvv = ((tgvvvv - einsum('ad,bc->acbd', tffvv, iv)
                       - einsum('ad,bc->acbd', iv, tffvv))
               / einsum('ab,cd->acbd', n1vv, n1vv))
    fgoooo = fermitools.math.transform(tfgoooo, {0: uo.T, 1: uo.T,
                                                 2: uo.T, 3: uo.T})
    fgovov = fermitools.math.transform(tfgovov, {0: uo.T, 1: uv.T,
                                                 2: uo.T, 3: uv.T})
    fgvvvv = fermitools.math.transform(tfgvvvv, {0: uv.T, 1: uv.T,
                                                 2: uv.T, 3: uv.T})
    return {'o,o,o,o': fgoooo, 'o,v,o,v': fgovov, 'v,v,v,v': fgvvvv}


def fancy_mixed_interaction(fov, gooov, govvv, m1oo, m1vv):
    no, uo = scipy.linalg.eigh(m1oo)
    nv, uv = scipy.linalg.eigh(m1vv)
    n1oo = fermitools.math.broadcast_sum({2: no, 3: no}) - 1
    n1vv = fermitools.math.broadcast_sum({2: nv, 3: nv}) - 1
    io = numpy.eye(*uo.shape)
    iv = numpy.eye(*uv.shape)
    ioo = (+ einsum('ik,la->iakl', io, fov)
           - einsum('mlka,im->iakl', gooov, m1oo)
           + einsum('ilke,ae->iakl', gooov, m1vv))
    ivv = (- einsum('ac,id->iadc', iv, fov)
           + einsum('mcad,im->iadc', govvv, m1oo)
           - einsum('iced,ae->iadc', govvv, m1vv))
    tfioo = fermitools.math.transform(ioo, {2: uo, 3: uo}) / n1oo
    tfivv = fermitools.math.transform(ivv, {2: uv, 3: uv}) / n1vv
    fioo = fermitools.math.transform(tfioo, {2: uo.T, 3: uo.T})
    fivv = fermitools.math.transform(tfivv, {2: uv.T, 3: uv.T})
    return {'o,o': fioo, 'v,v': fivv}


def amplitude_hessian_diag(ffoo, ffvv, goooo, govov, gvvvv, fgoooo, fgovov,
                           fgvvvv, t2):
    return (
        + cepa_amplitude_hessian(foo=+ffoo, fvv=-ffvv, goooo=goooo,
                                 govov=govov, gvvvv=gvvvv)
        + asym('2/3|6/7')(
              einsum('afec,ijeb,klfd->ijabklcd', fgvvvv, t2, t2))
        + asym('2/3|4/5')(
              einsum('kame,ijeb,mlcd->ijabklcd', fgovov, t2, t2))
        + asym('0/1|6/7')(
              einsum('meic,mjab,kled->ijabklcd', fgovov, t2, t2))
        + asym('0/1|4/5')(
              einsum('mkin,mjab,nlcd->ijabklcd', fgoooo, t2, t2)))


def amplitude_hessian_offd(fgoooo, fgovov,  fgvvvv, t2):
    return (
        + asym('2/3|6/7')(
               einsum('acef,ijeb,klfd->ijabklcd', fgvvvv, t2, t2))
        + asym('2/3|4/5')(
               einsum('nake,ijeb,nlcd->ijabklcd', fgovov, t2, t2))
        + asym('0/1|6/7')(
               einsum('mcif,mjab,klfd->ijabklcd', fgovov, t2, t2))
        + asym('0/1|4/5')(
               einsum('mnik,mjab,nlcd->ijabklcd', fgoooo, t2, t2)))


def mixed_hessian_diag(gooov, govvv, fioo, fivv, t2):
    no, _, nv, _ = t2.shape
    io = numpy.eye(no)
    iv = numpy.eye(nv)
    return (
        + asym('2/3')(
               einsum('ik,lacd->iaklcd', io, govvv))
        + asym('4/5')(
               einsum('ac,klid->iaklcd', iv, gooov))
        + asym('2/3')(
               einsum('iakm,mlcd->iaklcd', fioo, t2))
        + asym('4/5')(
               einsum('iaec,kled->iaklcd', fivv, t2))
        + asym('2/3|4/5')(
               einsum('ik,mcae,mled->iaklcd', io, govvv, t2))
        + asym('2/3|4/5')(
               einsum('ac,imke,mled->iaklcd', iv, gooov, t2))
        + 1./2 * asym('2/3')(
               einsum('ik,mnla,mncd->iaklcd', io, gooov, t2))
        + 1./2 * asym('4/5')(
               einsum('ac,idef,klef->iaklcd', iv, govvv, t2)))


def mixed_hessian_offd(gooov, govvv, fioo, fivv, t2):
    return (
        + asym('2/3')(
               einsum('iamk,mlcd->iaklcd', fioo, t2))
        + asym('4/5')(
               einsum('iace,kled->iaklcd', fivv, t2))
        + asym('2/3|4/5')(
               einsum('lead,kice->iaklcd', govvv, t2))
        + asym('2/3|4/5')(
               einsum('ilmd,kmca->iaklcd', gooov, t2))
        - einsum('klma,micd->iaklcd', gooov, t2)
        - einsum('iecd,klea->iaklcd', govvv, t2))


# Sigma vectors
def amplitude_hessian_diag_sigma(ffoo, ffvv, goooo, govov, gvvvv, fgoooo,
                                 fgovov, fgvvvv, t2):
    cepa_sigmaf = cepa_amplitude_hessian_sigma(
            foo=+ffoo, fvv=-ffvv, goooo=goooo, govov=govov, gvvvv=gvvvv)

    def _sigma(r2):
        return (
            + cepa_sigmaf(r2)
            + 1./2 * asym('2/3')(
                einsum('afec,ijeb,klfd,klcd...->ijab...', fgvvvv, t2, t2, r2))
            + 1./2 * asym('2/3')(
                einsum('kame,ijeb,mlcd,klcd...->ijab...', fgovov, t2, t2, r2))
            + 1./2 * asym('0/1')(
                einsum('meic,mjab,kled,klcd...->ijab...', fgovov, t2, t2, r2))
            + 1./2 * asym('0/1')(
                einsum('mkin,mjab,nlcd,klcd...->ijab...', fgoooo, t2, t2, r2)))

    return _sigma


def amplitude_hessian_offd_sigma(fgoooo, fgovov, fgvvvv, t2):

    def _sigma(r2):
        return (
            + 1./2 * asym('2/3')(
                einsum('acef,ijeb,klfd,klcd...->ijab...', fgvvvv, t2, t2, r2))
            + 1./2 * asym('2/3')(
                einsum('nake,ijeb,nlcd,klcd...->ijab...', fgovov, t2, t2, r2))
            + 1./2 * asym('0/1')(
                einsum('mcif,mjab,klfd,klcd...->ijab...', fgovov, t2, t2, r2))
            + 1./2 * asym('0/1')(
                einsum('mnik,mjab,nlcd,klcd...->ijab...', fgoooo, t2, t2, r2)))

    return _sigma


def mixed_hessian_right_diag_sigma(gooov, govvv, fioo, fivv, t2):

    def _sigma(r2):
        return (
            + 1./2 * einsum('lacd,ilcd...->ia...', govvv, r2)
            + 1./2 * einsum('klid,klad...->ia...', gooov, r2)
            + 1./2 * einsum('iakm,mlcd,klcd...->ia...', fioo, t2, r2)
            + 1./2 * einsum('iaec,kled,klcd...->ia...', fivv, t2, r2)
            + einsum('mcae,mled,ilcd...->ia...', govvv, t2, r2)
            + einsum('imke,mled,klad...->ia...', gooov, t2, r2)
            + 1./4 * einsum('mnla,mncd,ilcd...->ia...', gooov, t2, r2)
            + 1./4 * einsum('idef,klef,klad...->ia...', govvv, t2, r2))

    return _sigma


def mixed_hessian_right_offd_sigma(gooov, govvv, fioo, fivv, t2):

    def _sigma(r2):
        return (
            + 1./2 * einsum('iamk,mlcd,klcd...->ia...', fioo, t2, r2)
            + 1./2 * einsum('iace,kled,klcd...->ia...', fivv, t2, r2)
            + einsum('lead,kice,klcd...->ia...', govvv, t2, r2)
            + einsum('ilmd,kmca,klcd...->ia...', gooov, t2, r2)
            - 1./4 * einsum('klma,micd,klcd...->ia...', gooov, t2, r2)
            - 1./4 * einsum('iecd,klea,klcd...->ia...', govvv, t2, r2))

    return _sigma


def mixed_hessian_left_diag_sigma(gooov, govvv, fioo, fivv, t2):

    def _sigma(r1):
        return (
            + asym('0/1')(
                   einsum('jcab,ic...->ijab...', govvv, r1))
            + asym('2/3')(
                   einsum('ijkb,ka...->ijab...', gooov, r1))
            + asym('0/1')(
                   einsum('kcim,mjab,kc...->ijab...', fioo, t2, r1))
            + asym('2/3')(
                   einsum('kcea,ijeb,kc...->ijab...', fivv, t2, r1))
            + asym('0/1|2/3')(
                   einsum('mace,mjeb,ic...->ijab...', govvv, t2, r1))
            + asym('0/1|2/3')(
                   einsum('kmie,mjeb,ka...->ijab...', gooov, t2, r1))
            + 1./2 * asym('0/1')(
                   einsum('mnjc,mnab,ic...->ijab...', gooov, t2, r1))
            + 1./2 * asym('2/3')(
                   einsum('kbef,ijef,ka...->ijab...', govvv, t2, r1)))

    return _sigma


def mixed_hessian_left_offd_sigma(gooov, govvv, fioo, fivv, t2):

    def _sigma(r1):
        return (
            + asym('0/1')(
                   einsum('kcmi,mjab,kc...->ijab...', fioo, t2, r1))
            + asym('2/3')(
                   einsum('kcae,ijeb,kc...->ijab...', fivv, t2, r1))
            + asym('0/1|2/3')(
                   einsum('jecb,ikae,kc...->ijab...', govvv, t2, r1))
            + asym('0/1|2/3')(
                   einsum('kjmb,imac,kc...->ijab...', gooov, t2, r1))
            - einsum('ijmc,mkab,kc...->ijab...', gooov, t2, r1)
            - einsum('keab,ijec,kc...->ijab...', govvv, t2, r1))

    return _sigma


def main():
    from scripts import odc12

    CHARGE = +0
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
    p_ao = interface.integrals.dipole(BASIS, LABELS, COORDS)
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

    # Solve OCEPA0
    en_nuc = fermitools.chem.nuc.energy(labels=LABELS, coords=COORDS)
    t2_guess = numpy.zeros((nocc, nocc, norb-nocc, norb-nocc))
    en_elec, c, t2 = odc12.solve(
            norb=norb, nocc=nocc, h_aso=h_aso, g_aso=g_aso, c_guess=c,
            t2_guess=t2_guess, niter=200, e_thresh=1e-14, r_thresh=1e-12,
            print_conv=True)
    en_tot = en_elec + en_nuc
    print("\nGround state energy:")
    print('{:20.15f}'.format(en_tot))

    # Build blocks of the electronic Hessian
    o = slice(None, nocc)
    v = slice(nocc, None)
    no = nocc
    nv = norb - nocc
    nsingles = no * nv
    ndoubles = no * (no - 1) * nv * (nv - 1) // 4
    h = fermitools.math.transform(h_aso, {0: c, 1: c})
    g = fermitools.math.transform(g_aso, {0: c, 1: c, 2: c, 3: c})
    dm1oo = numpy.eye(no)
    dm1vv = numpy.zeros((nv, nv))
    cm1oo, cm1vv = fermitools.oo.odc12.onebody_correlation_density(t2)
    dm1 = scipy.linalg.block_diag(dm1oo, dm1vv)
    cm1 = scipy.linalg.block_diag(cm1oo, cm1vv)
    m1 = dm1 + cm1
    k2 = doubles_cumulant(t2)
    m2 = doubles_density(m1, k2)

    foo = fermitools.oo.fock_block(
            hxy=h[o, o], goxoy=g[o, o, o, o], m1oo=m1[o, o],
            gxvyv=g[o, v, o, v], m1vv=m1[v, v])
    fov = fermitools.oo.fock_block(
            hxy=h[o, v], goxoy=g[o, o, o, v], m1oo=m1[o, o],
            gxvyv=g[o, v, v, v], m1vv=m1[v, v])
    fvv = fermitools.oo.fock_block(
            hxy=h[v, v], goxoy=g[o, v, o, v], m1oo=m1[o, o],
            gxvyv=g[v, v, v, v], m1vv=m1[v, v])
    ffoo = fermitools.oo.odc12.fancy_property(foo, m1[o, o])
    ffvv = fermitools.oo.odc12.fancy_property(fvv, m1[v, v])
    fg = fancy_repulsion(
            ffoo, ffvv, g[o, o, o, o], g[o, v, o, v], g[v, v, v, v],
            m1[o, o], m1[v, v])
    fi = fancy_mixed_interaction(
            fov, g[o, o, o, v], g[o, v, v, v], m1[o, o], m1[v, v])

    # Raveling operators
    v_orb_raveler = fermitools.math.raveler({0: (0, 1)})
    v_amp_raveler = fermitools.math.asym.megaraveler({0: ((0, 1), (2, 3))})
    m_orb_raveler = fermitools.math.raveler({0: (0, 1), 1: (2, 3)})
    m_mix_raveler = fermitools.math.asym.megaraveler(
            {0: ((0,), (1,)), 1: ((2, 3), (4, 5))})
    m_amp_raveler = fermitools.math.asym.megaraveler(
            {0: ((0, 1), (2, 3)), 1: ((4, 5), (6, 7))})

    def v_orb_unraveler(r1):
        shape = (no, nv) if r1.ndim == 1 else (no, nv) + r1.shape[1:]
        return numpy.reshape(r1, shape)

    def v_amp_unraveler(r2):
        noo = no * (no - 1) // 2
        nvv = nv * (nv - 1) // 2
        shape = (noo, nvv) if r2.ndim == 1 else (noo, nvv) + r2.shape[1:]
        unravf = fermitools.math.asym.unraveler({0: (0, 1), 1: (2, 3)})
        return unravf(numpy.reshape(r2, shape))

    def add(f, g):

        def _sum(*args, **kwargs):
            return f(*args, **kwargs) + g(*args, **kwargs)

        return _sum

    def sub(f, g):

        def _diff(*args, **kwargs):
            return f(*args, **kwargs) - g(*args, **kwargs)

        return _diff

    a_orb = m_orb_raveler(orbital_hessian_diag(
            h[o, o], h[v, v], g[o, o, o, o], g[o, o, v, v], g[o, v, o, v],
            g[v, v, v, v], m1[o, o], m1[v, v], m2[o, o, o, o], m2[o, o, v, v],
            m2[o, v, o, v], m2[v, v, v, v]))
    b_orb = m_orb_raveler(orbital_hessian_offd(
            g[o, o, o, o], g[o, o, v, v], g[o, v, o, v], g[v, v, v, v],
            m2[o, o, o, o], m2[o, o, v, v], m2[o, v, o, v], m2[v, v, v, v]))
    s_orb = m_orb_raveler(orbital_metric(m1[o, o], m1[v, v]))

    a_mix = m_mix_raveler(mixed_hessian_diag(
            g[o, o, o, v], g[o, v, v, v], fi['o,o'], fi['v,v'], t2))
    b_mix = m_mix_raveler(mixed_hessian_offd(
            g[o, o, o, v], g[o, v, v, v], fi['o,o'], fi['v,v'], t2))

    a_amp = m_amp_raveler(amplitude_hessian_diag(
            ffoo, ffvv, g[o, o, o, o], g[o, v, o, v], g[v, v, v, v],
            fg['o,o,o,o'], fg['o,v,o,v'], fg['v,v,v,v'], t2))
    b_amp = m_amp_raveler(amplitude_hessian_offd(
            fg['o,o,o,o'], fg['o,v,o,v'], fg['v,v,v,v'], t2))
    s_amp = numpy.eye(*a_amp.shape)

    # Evaluate the excitation energies
    a = numpy.bmat([[a_orb, a_mix], [a_mix.T, a_amp]])
    b = numpy.bmat([[b_orb, b_mix], [b_mix.T, b_amp]])
    s = scipy.linalg.block_diag(s_orb, s_amp)

    e = numpy.bmat([[a, b], [b, a]])
    m = scipy.linalg.block_diag(s, -s)
    w_old = numpy.real(sorted(scipy.linalg.eigvals(e, b=m)))
    print("\nSpectrum:")
    print(w_old)

    # Evaluate dipole polarizability using linear response theory
    p_aso = fermitools.math.spinorb.expand(p_ao, brakets=((1, 2),))
    p = fermitools.math.transform(p_aso, {1: c, 2: c})
    fpoo = fermitools.oo.odc12.fancy_property(p[:, o, o], m1[o, o])
    fpvv = fermitools.oo.odc12.fancy_property(p[:, v, v], m1[v, v])
    t_orb = v_orb_raveler(orbital_property_gradient(
            p[:, o, v], m1[o, o], m1[v, v]))
    t_amp = v_amp_raveler(amplitude_property_gradient(fpoo, -fpvv, t2))
    t = numpy.concatenate((t_orb, t_amp), axis=0)
    r = static_response_vector(a, b, t)
    alpha_old = static_linear_response_function(t, r)
    print(alpha_old.round(8))

    # Test sigma vectors
    sig_a_orb = orbital_hessian_diag_sigma(
            h[o, o], h[v, v], g[o, o, o, o], g[o, o, v, v], g[o, v, o, v],
            g[v, v, v, v], m1[o, o], m1[v, v], m2[o, o, o, o], m2[o, o, v, v],
            m2[o, v, o, v], m2[v, v, v, v])
    sig_b_orb = orbital_hessian_offd_sigma(
            g[o, o, o, o], g[o, o, v, v], g[o, v, o, v], g[v, v, v, v],
            m2[o, o, o, o], m2[o, o, v, v], m2[o, v, o, v], m2[v, v, v, v])
    s_orb_inv = scipy.linalg.inv(s_orb)
    sig_s_orb_inv = scipy.sparse.linalg.aslinearoperator(s_orb_inv)
    rsig_a_mix = mixed_hessian_right_diag_sigma(
            g[o, o, o, v], g[o, v, v, v], fi['o,o'], fi['v,v'], t2)
    rsig_b_mix = mixed_hessian_right_offd_sigma(
            g[o, o, o, v], g[o, v, v, v], fi['o,o'], fi['v,v'], t2)
    lsig_a_mix = mixed_hessian_left_diag_sigma(
            g[o, o, o, v], g[o, v, v, v], fi['o,o'], fi['v,v'], t2)
    lsig_b_mix = mixed_hessian_left_offd_sigma(
            g[o, o, o, v], g[o, v, v, v], fi['o,o'], fi['v,v'], t2)
    sig_a_amp = amplitude_hessian_diag_sigma(
            ffoo, ffvv, g[o, o, o, o], g[o, v, o, v], g[v, v, v, v],
            fg['o,o,o,o'], fg['o,v,o,v'], fg['v,v,v,v'], t2)
    sig_b_amp = amplitude_hessian_offd_sigma(
            fg['o,o,o,o'], fg['o,v,o,v'], fg['v,v,v,v'], t2)

    sig_e_orb_sum = functoolz.compose(
            v_orb_raveler, add(sig_a_orb, sig_b_orb), v_orb_unraveler)
    sig_e_orb_diff = functoolz.compose(
            v_orb_raveler, sub(sig_a_orb, sig_b_orb), v_orb_unraveler)
    rsig_e_mix_sum = functoolz.compose(
            v_orb_raveler, add(rsig_a_mix, rsig_b_mix), v_amp_unraveler)
    rsig_e_mix_diff = functoolz.compose(
            v_orb_raveler, sub(rsig_a_mix, rsig_b_mix), v_amp_unraveler)
    lsig_e_mix_sum = functoolz.compose(
            v_amp_raveler, add(lsig_a_mix, lsig_b_mix), v_orb_unraveler)
    lsig_e_mix_diff = functoolz.compose(
            v_amp_raveler, sub(lsig_a_mix, lsig_b_mix), v_orb_unraveler)
    sig_e_amp_sum = functoolz.compose(
            v_amp_raveler, add(sig_a_amp, sig_b_amp), v_amp_unraveler)
    sig_e_amp_diff = functoolz.compose(
            v_amp_raveler, sub(sig_a_amp, sig_b_amp), v_amp_unraveler)

    from numpy.testing import assert_almost_equal

    i1 = numpy.eye(nsingles)
    i2 = numpy.eye(ndoubles)
    assert_almost_equal(sig_e_orb_sum(i1), a_orb + b_orb)
    assert_almost_equal(sig_e_orb_diff(i1), a_orb - b_orb)
    assert_almost_equal(rsig_e_mix_sum(i2), a_mix + b_mix)
    assert_almost_equal(rsig_e_mix_diff(i2), a_mix - b_mix)
    assert_almost_equal(lsig_e_mix_sum(i1), numpy.transpose(a_mix + b_mix))
    assert_almost_equal(lsig_e_mix_diff(i1), numpy.transpose(a_mix - b_mix))
    assert_almost_equal(sig_e_amp_sum(i2), a_amp + b_amp)
    assert_almost_equal(sig_e_amp_diff(i2), a_amp - b_amp)

    sig_e_sum = combined_hessian_sigma(
            nsingles, sig_e_orb_sum, rsig_e_mix_sum, lsig_e_mix_sum,
            sig_e_amp_sum)
    sig_e_diff = combined_hessian_sigma(
            nsingles, sig_e_orb_diff, rsig_e_mix_diff, lsig_e_mix_diff,
            sig_e_amp_diff)
    sig_s_inv = combined_metric_sigma(nsingles, sig_s_orb_inv)
    sig_e_eff = effective_response_hamiltonian_sigma(
            sig_e_sum, sig_e_diff, sig_s_inv)

    dim = nsingles + ndoubles
    e_ = scipy.sparse.linalg.LinearOperator((dim, dim), matvec=sig_e_eff)
    w2, u = scipy.sparse.linalg.eigs(e_, k=dim-2, which='SR')
    w = numpy.sqrt(numpy.real(sorted(w2)))
    print(w)
    print(w / w_old[dim:2*dim-2])

    # Response function
    r = solve_static_response_vector(dim, sig_e_sum, t)
    alpha = numpy.tensordot(r, t, axes=(0, 0))
    print(alpha.round(10))
    print(numpy.diag(alpha) / numpy.diag(alpha_old))


if __name__ == '__main__':
    main()

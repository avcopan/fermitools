import numpy
import scipy.linalg as spla

import fermitools
from fermitools.math import einsum
from fermitools.math.asym import antisymmetrizer_product as asym

import interfaces.psi4 as interface
from .odc12 import fock
from .odc12 import singles_reference_density
from .odc12 import singles_correlation_density
from .odc12 import doubles_cumulant
from .odc12 import doubles_density
from .lr_ocepa0 import diagonal_orbital_hessian
from .lr_ocepa0 import offdiagonal_orbital_hessian
from .lr_ocepa0 import (diagonal_amplitude_hessian as
                        cepa_diagonal_amplitude_hessian)
from .lr_ocepa0 import orbital_property_gradient
from .lr_ocepa0 import amplitude_property_gradient
from .lr_ocepa0 import static_response_vector
from .lr_ocepa0 import static_linear_response_function
from .lr_ocepa0 import diagonal_orbital_metric


def fancy_property(poo, pvv, m1oo, m1vv):
    '''
    ff_ov and ff_vo are undefined, so return a dictionary with the diagonal
    blocks.
    '''
    no, uo = spla.eigh(m1oo)
    nv, uv = spla.eigh(m1vv)
    ax1, ax2 = poo.ndim - 2, poo.ndim - 1
    n1oo = fermitools.math.broadcast_sum({ax1: no, ax2: no}) - 1
    n1vv = fermitools.math.broadcast_sum({ax1: nv, ax2: nv}) - 1
    tfpoo = fermitools.math.transform(poo, {ax1: uo, ax2: uo}) / n1oo
    tfpvv = fermitools.math.transform(pvv, {ax1: uv, ax2: uv}) / n1vv
    fpoo = fermitools.math.transform(tfpoo, {ax1: uo.T, ax2: uo.T})
    fpvv = fermitools.math.transform(tfpvv, {ax1: uv.T, ax2: uv.T})
    return {'o,o': fpoo, 'v,v': fpvv}


def fancy_repulsion(ffoo, ffvv, goooo, govov, gvvvv, m1oo, m1vv):
    no, uo = spla.eigh(m1oo)
    nv, uv = spla.eigh(m1vv)
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
    no, uo = spla.eigh(m1oo)
    nv, uv = spla.eigh(m1vv)
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


def diagonal_amplitude_hessian(ffoo, ffvv, goooo, govov, gvvvv,
                               fgoooo, fgovov, fgvvvv, t2):
    a_cepa = cepa_diagonal_amplitude_hessian(foo=+ffoo, fvv=-ffvv,
                                             goooo=goooo, govov=govov,
                                             gvvvv=gvvvv)
    a_dc = (+ asym('2/3|6/7')(
                  einsum('afec,ijeb,klfd->ijabklcd', fgvvvv, t2, t2))
            + asym('2/3|4/5')(
                  einsum('kame,ijeb,mlcd->ijabklcd', fgovov, t2, t2))
            + asym('0/1|6/7')(
                  einsum('meic,mjab,kled->ijabklcd', fgovov, t2, t2))
            + asym('0/1|4/5')(
                  einsum('mkin,mjab,nlcd->ijabklcd', fgoooo, t2, t2))
            )
    a_dc_cmp = fermitools.math.asym.compound_index(a_dc,
                                                   {0: (0, 1), 1: (2, 3),
                                                    2: (4, 5), 3: (6, 7)})
    return a_cepa + numpy.reshape(a_dc_cmp, a_cepa.shape)


def offdiagonal_amplitude_hessian(fgoooo, fgovov,  fgvvvv, t2):
    no, nv, _, _ = fgovov.shape
    ndoubles = no * (no - 1) * nv * (nv - 1) // 4
    b = (+ asym('2/3|6/7')(
                einsum('acef,ijeb,klfd->ijabklcd', fgvvvv, t2, t2))
         + asym('2/3|4/5')(
                einsum('nake,ijeb,nlcd->ijabklcd', fgovov, t2, t2))
         + asym('0/1|6/7')(
                einsum('mcif,mjab,klfd->ijabklcd', fgovov, t2, t2))
         + asym('0/1|4/5')(
                einsum('mnik,mjab,nlcd->ijabklcd', fgoooo, t2, t2)))
    b_cmp = fermitools.math.asym.compound_index(b, {0: (0, 1), 1: (2, 3),
                                                    2: (4, 5), 3: (6, 7)})
    return numpy.reshape(b_cmp, (ndoubles, ndoubles))


def diagonal_mixed_hessian(gooov, govvv, fioo, fivv, t2):
    no, _, nv, _ = t2.shape
    nsingles = no * nv
    ndoubles = no * (no - 1) * nv * (nv - 1) // 4
    io = numpy.eye(no)
    iv = numpy.eye(nv)
    a = (+ asym('2/3')(
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
    a_cmp = fermitools.math.asym.compound_index(a, {2: (2, 3), 3: (4, 5)})
    return numpy.reshape(a_cmp, (nsingles, ndoubles))


def offdiagonal_mixed_hessian(gooov, govvv, fioo, fivv, t2):
    no, _, nv, _ = t2.shape
    nsingles = no * nv
    ndoubles = no * (no - 1) * nv * (nv - 1) // 4
    b = (+ asym('2/3')(
                einsum('iamk,mlcd->iaklcd', fioo, t2))
         + asym('4/5')(
                einsum('iace,kled->iaklcd', fivv, t2))
         + asym('2/3|4/5')(
                einsum('lead,kice->iaklcd', govvv, t2))
         + asym('2/3|4/5')(
                einsum('ilmd,kmca->iaklcd', gooov, t2))
         - einsum('klma,micd->iaklcd', gooov, t2)
         - einsum('iecd,klea->iaklcd', govvv, t2))
    b_cmp = fermitools.math.asym.compound_index(b, {2: (2, 3), 3: (4, 5)})
    return numpy.reshape(b_cmp, (nsingles, ndoubles))


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
    en_nuc = fermitools.chem.nuc.energy(labels=LABELS, coords=COORDS)
    t2_guess = numpy.zeros((nocc, nocc, norb-nocc, norb-nocc))
    en_elec, c, t2 = odc12.solve(norb=norb, nocc=nocc, h_aso=h_aso,
                                 g_aso=g_aso, c_guess=c,
                                 t2_guess=t2_guess, niter=200,
                                 e_thresh=1e-14, r_thresh=1e-12,
                                 print_conv=True)
    en_tot = en_elec + en_nuc
    print("Total energy:")
    print('{:20.15f}'.format(en_tot))

    # Build blocks of the electronic Hessian
    h = fermitools.math.transform(h_aso, {0: c, 1: c})
    g = fermitools.math.transform(g_aso, {0: c, 1: c, 2: c, 3: c})
    m1_ref = singles_reference_density(norb=norb, nocc=nocc)
    m1_cor = singles_correlation_density(t2)
    m1 = m1_ref + m1_cor
    k2 = doubles_cumulant(t2)
    m2 = doubles_density(m1, k2)

    f = fock(h, g, m1)
    o = slice(None, nocc)
    v = slice(nocc, None)
    ff = odc12.fancy_fock(f[o, o], f[v, v], m1[o, o], m1[v, v])
    fg = fancy_repulsion(ff['o,o'], ff['v,v'], g[o, o, o, o], g[o, v, o, v],
                         g[v, v, v, v], m1[o, o], m1[v, v])
    fi = fancy_mixed_interaction(f[o, v], g[o, o, o, v], g[o, v, v, v],
                                 m1[o, o], m1[v, v])

    a_orb = diagonal_orbital_hessian(h[o, o], h[v, v], g[o, o, o, o],
                                     g[o, o, v, v], g[o, v, o, v],
                                     g[v, v, v, v], m1[o, o], m1[v, v],
                                     m2[o, o, o, o], m2[o, o, v, v],
                                     m2[o, v, o, v], m2[v, v, v, v])
    a_mix = diagonal_mixed_hessian(g[o, o, o, v], g[o, v, v, v],
                                   fi['o,o'], fi['v,v'], t2)
    a_amp = diagonal_amplitude_hessian(ff['o,o'], ff['v,v'], g[o, o, o, o],
                                       g[o, v, o, v], g[v, v, v, v],
                                       fg['o,o,o,o'], fg['o,v,o,v'],
                                       fg['v,v,v,v'], t2)
    b_orb = offdiagonal_orbital_hessian(g[o, o, o, o], g[o, o, v, v],
                                        g[o, v, o, v], g[v, v, v, v],
                                        m2[o, o, o, o], m2[o, o, v, v],
                                        m2[o, v, o, v], m2[v, v, v, v])
    b_mix = offdiagonal_mixed_hessian(g[o, o, o, v], g[o, v, v, v],
                                      fi['o,o'], fi['v,v'], t2)
    b_amp = offdiagonal_amplitude_hessian(fg['o,o,o,o'], fg['o,v,o,v'],
                                          fg['v,v,v,v'], t2)
    print(a_mix.shape)
    print(b_mix.shape)

    # Evaluate dipole polarizability using linear response theory
    p_ao = interface.integrals.dipole(BASIS, LABELS, COORDS)
    p_aso = fermitools.math.spinorb.expand(p_ao, brakets=((1, 2),))
    p = fermitools.math.transform(p_aso, {1: c, 2: c})
    fp = fancy_property(p[:, o, o], p[:, v, v], m1[o, o], m1[v, v])
    t_orb = orbital_property_gradient(p[:, o, v], m1[o, o], m1[v, v])
    t_amp = amplitude_property_gradient(fp['o,o'], -fp['v,v'], t2)

    a = numpy.bmat([[a_orb, -a_mix], [-a_mix.T, a_amp]])
    b = numpy.bmat([[b_orb, -b_mix], [-b_mix.T, b_amp]])
    t = numpy.bmat([[t_orb], [t_amp]])
    r = static_response_vector(a, b, t)
    alpha = static_linear_response_function(t, r)

    print(numpy.real(alpha).round(8))

    # Evaluate the excitation energies
    s_orb = diagonal_orbital_metric(m1[o, o], m1[v, v])
    s_amp = numpy.eye(*a_amp.shape)
    s = spla.block_diag(s_orb, s_amp)

    e = numpy.bmat([[a, b], [b, a]])
    m = spla.block_diag(s, -s)
    w = numpy.real(spla.eigvals(e, b=m))
    print(numpy.array(sorted(w)))


if __name__ == '__main__':
    main()

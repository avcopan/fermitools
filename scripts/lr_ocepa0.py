import numpy

import fermitools
from fermitools.math import einsum
from fermitools.math.asym import antisymmetrizer_product as asym

import interfaces.psi4 as interface
from .lr_scf import orbital_hessian_diag
from .lr_scf import orbital_hessian_offd
from .lr_scf import orbital_property_gradient
from .lr_scf import orbital_metric
from .lr_scf import static_response_vector
from .lr_scf import static_linear_response_function
from .lr_scf import solve_static_response_vector
from .lr_scf import effective_response_hamiltonian_sigma

# sigmas
from .lr_scf import orbital_hessian_sum_sigma
from .lr_scf import orbital_hessian_diff_sigma


def amplitude_hessian(foo, fvv, goooo, govov, gvvvv):
    no, nv, _, _ = govov.shape
    ndoubles = no * (no - 1) * nv * (nv - 1) // 4
    io = numpy.eye(no)
    iv = numpy.eye(nv)
    a = (+ asym('2/3|4/5|6/7')(
               einsum('ik,jl,ac,bd->ijabklcd', io, io, fvv, iv))
         - asym('0/1|4/5|6/7')(
               einsum('ik,jl,ac,bd->ijabklcd', foo, io, iv, iv))
         + asym('4/5')(
               einsum('ik,jl,abcd->ijabklcd', io, io, gvvvv))
         + asym('6/7')(
               einsum('ijkl,ac,bd->ijabklcd', goooo, iv, iv))
         - asym('0/1|2/3|4/5|6/7')(
               einsum('ik,jcla,bd->ijabklcd', io, govov, iv)))
    a_cmp = fermitools.math.asym.compound_index(a, {0: (0, 1), 1: (2, 3),
                                                    2: (4, 5), 3: (6, 7)})
    return numpy.reshape(a_cmp, (ndoubles, ndoubles))


def mixed_interaction(fov, gooov, govvv):
    no, nv, _, _ = govvv.shape
    io = numpy.eye(no)
    iv = numpy.eye(nv)
    ioo = (+ einsum('ik,la->iakl', io, fov)
           - einsum('ilka->iakl', gooov))
    ivv = (- einsum('ac,id->iadc', iv, fov)
           + einsum('icad->iadc', govvv))
    return {'o,o': ioo, 'v,v': ivv}


def mixed_hessian_diag(fov, gooov, govvv, t2):
    no, _, nv, _ = t2.shape
    nsingles = no * nv
    ndoubles = no * (no - 1) * nv * (nv - 1) // 4
    io = numpy.eye(no)
    iv = numpy.eye(nv)
    i = mixed_interaction(fov, gooov, govvv)
    a = (+ asym('2/3')(
                einsum('ik,lacd->iaklcd', io, govvv))
         + asym('4/5')(
                einsum('ac,klid->iaklcd', iv, gooov))
         + asym('2/3')(
                einsum('iakm,mlcd->iaklcd', i['o,o'], t2))
         - asym('4/5')(
                einsum('iaec,kled->iaklcd', i['v,v'], t2))
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


def mixed_hessian_offd(fov, gooov, govvv, t2):
    no, _, nv, _ = t2.shape
    nsingles = no * nv
    ndoubles = no * (no - 1) * nv * (nv - 1) // 4
    i = mixed_interaction(fov, gooov, govvv)
    b = (+ asym('2/3')(
                einsum('iamk,mlcd->iaklcd', i['o,o'], t2))
         - asym('4/5')(
                einsum('iace,kled->iaklcd', i['v,v'], t2))
         + asym('2/3|4/5')(
                einsum('lead,kice->iaklcd', govvv, t2))
         + asym('2/3|4/5')(
                einsum('ilmd,kmca->iaklcd', gooov, t2))
         - einsum('klma,micd->iaklcd', gooov, t2)
         - einsum('iecd,klea->iaklcd', govvv, t2))
    b_cmp = fermitools.math.asym.compound_index(b, {2: (2, 3), 3: (4, 5)})
    return numpy.reshape(b_cmp, (nsingles, ndoubles))


def amplitude_property_gradient(poo, pvv, t2):
    no, _, nv, _ = t2.shape
    ndoubles = no * (no - 1) * nv * (nv - 1) // 4
    t = (+ asym('2/3')(
               einsum('...ac,ijcb->ijab...', pvv, t2))
         - asym('0/1')(
               einsum('...ik,kjab->ijab...', poo, t2)))
    t_cmp = fermitools.math.asym.compound_index(t, {0: (0, 1), 1: (2, 3)})
    shape = (ndoubles,) + t.shape[4:]
    return numpy.reshape(t_cmp, shape)


# Sigma vectors
def amplitude_hessian_sigma(foo, fvv, goooo, govov, gvvvv):
    no, nv, _, _ = govov.shape
    noo = no * (no - 1) // 2
    nvv = nv * (nv - 1) // 2
    ndoubles = noo * nvv

    def _sigma(r2_flat):
        cols = 1 if r2_flat.ndim is 1 else r2_flat.shape[1]
        r2_cmp = numpy.reshape(r2_flat, (noo, nvv, cols))
        r2 = fermitools.math.asym.unravel_compound_index(r2_cmp, {0: (0, 1),
                                                                  1: (2, 3)})
        ar2 = (+ asym('2/3')(einsum('ac,ijcbx->ijabx', fvv, r2))
               - asym('0/1')(einsum('ik,kjabx->ijabx', foo, r2))
               + 1./2 * einsum('abcd,ijcdx->ijabx', gvvvv, r2)
               + 1./2 * einsum('ijkl,klabx->ijabx', goooo, r2)
               - asym('0/1|2/3')(einsum('jcla,ilcbx->ijabx', govov, r2))
               )
        ar2_cmp = fermitools.math.asym.compound_index(ar2, {0: (0, 1),
                                                            1: (2, 3)})
        return numpy.squeeze(numpy.reshape(ar2_cmp, (ndoubles, cols)))

    return _sigma


def mixed_hessian_right_sum_sigma(fov, gooov, govvv, t2):
    no, _, nv, _ = t2.shape
    nsingles = no * nv
    noo = no * (no - 1) // 2
    nvv = nv * (nv - 1) // 2
    i = mixed_interaction(fov, gooov, govvv)

    def _sigma(r2_flat):
        cols = 1 if r2_flat.ndim is 1 else r2_flat.shape[1]
        r2_cmp = numpy.reshape(r2_flat, (noo, nvv, cols))
        r2 = fermitools.math.asym.unravel_compound_index(r2_cmp, {0: (0, 1),
                                                                  1: (2, 3)})
        ar2 = (
            # a terms
            + 1./2 * einsum('lacd,ilcdx->iax', govvv, r2)
            + 1./2 * einsum('klid,kladx->iax', gooov, r2)
            + 1./2 * einsum('iakm,mlcd,klcdx->iax', i['o,o'], t2, r2)
            - 1./2 * einsum('iaec,kled,klcdx->iax', i['v,v'], t2, r2)
            + einsum('mcae,mled,ilcdx->iax', govvv, t2, r2)
            + einsum('imke,mled,kladx->iax', gooov, t2, r2)
            + 1./4 * einsum('mnla,mncd,ilcdx->iax', gooov, t2, r2)
            + 1./4 * einsum('idef,klef,kladx->iax', govvv, t2, r2)
            # b terms
            + 1./2 * einsum('iamk,mlcd,klcdx->iax', i['o,o'], t2, r2)
            - 1./2 * einsum('iace,kled,klcdx->iax', i['v,v'], t2, r2)
            + einsum('lead,kice,klcdx->iax', govvv, t2, r2)
            + einsum('ilmd,kmca,klcdx->iax', gooov, t2, r2)
            - 1./4 * einsum('klma,micd,klcdx->iax', gooov, t2, r2)
            - 1./4 * einsum('iecd,klea,klcdx->iax', govvv, t2, r2))
        return numpy.squeeze(numpy.reshape(ar2, (nsingles, cols)))

    return _sigma


def mixed_hessian_right_diff_sigma(fov, gooov, govvv, t2):
    no, _, nv, _ = t2.shape
    nsingles = no * nv
    noo = no * (no - 1) // 2
    nvv = nv * (nv - 1) // 2
    i = mixed_interaction(fov, gooov, govvv)

    def _sigma(r2_flat):
        cols = 1 if r2_flat.ndim is 1 else r2_flat.shape[1]
        r2_cmp = numpy.reshape(r2_flat, (noo, nvv, cols))
        r2 = fermitools.math.asym.unravel_compound_index(r2_cmp, {0: (0, 1),
                                                                  1: (2, 3)})
        ar2 = (
            # a terms
            + 1./2 * einsum('lacd,ilcdx->iax', govvv, r2)
            + 1./2 * einsum('klid,kladx->iax', gooov, r2)
            + 1./2 * einsum('iakm,mlcd,klcdx->iax', i['o,o'], t2, r2)
            - 1./2 * einsum('iaec,kled,klcdx->iax', i['v,v'], t2, r2)
            + einsum('mcae,mled,ilcdx->iax', govvv, t2, r2)
            + einsum('imke,mled,kladx->iax', gooov, t2, r2)
            + 1./4 * einsum('mnla,mncd,ilcdx->iax', gooov, t2, r2)
            + 1./4 * einsum('idef,klef,kladx->iax', govvv, t2, r2)
            # b terms
            - 1./2 * einsum('iamk,mlcd,klcdx->iax', i['o,o'], t2, r2)
            + 1./2 * einsum('iace,kled,klcdx->iax', i['v,v'], t2, r2)
            - einsum('lead,kice,klcdx->iax', govvv, t2, r2)
            - einsum('ilmd,kmca,klcdx->iax', gooov, t2, r2)
            + 1./4 * einsum('klma,micd,klcdx->iax', gooov, t2, r2)
            + 1./4 * einsum('iecd,klea,klcdx->iax', govvv, t2, r2))
        return numpy.squeeze(numpy.reshape(ar2, (nsingles, cols)))

    return _sigma


def mixed_hessian_left_sum_sigma(fov, gooov, govvv, t2):
    no, _, nv, _ = t2.shape
    noo = no * (no - 1) // 2
    nvv = nv * (nv - 1) // 2
    ndoubles = noo * nvv
    i = mixed_interaction(fov, gooov, govvv)

    def _sigma(r1_flat):
        cols = 1 if r1_flat.ndim is 1 else r1_flat.shape[1]
        r1 = numpy.reshape(r1_flat, (no, nv, cols))
        ar1 = (
            # a terms
            + asym('0/1')(einsum('jcab,icx->ijabx', govvv, r1))
            + asym('2/3')(einsum('ijkb,kax->ijabx', gooov, r1))
            + asym('0/1')(einsum('kcim,mjab,kcx->ijabx', i['o,o'], t2, r1))
            - asym('2/3')(einsum('kcea,ijeb,kcx->ijabx', i['v,v'], t2, r1))
            + asym('0/1|2/3')(einsum('mace,mjeb,icx->ijabx', govvv, t2, r1))
            + asym('0/1|2/3')(einsum('kmie,mjeb,kax->ijabx', gooov, t2, r1))
            + 1./2 * asym('0/1')(einsum('mnjc,mnab,icx->ijabx', gooov, t2, r1))
            + 1./2 * asym('2/3')(einsum('kbef,ijef,kax->ijabx', govvv, t2, r1))
            # b terms
            + asym('0/1')(einsum('kcmi,mjab,kcx->ijabx', i['o,o'], t2, r1))
            - asym('2/3')(einsum('kcae,ijeb,kcx->ijabx', i['v,v'], t2, r1))
            + asym('0/1|2/3')(einsum('jecb,ikae,kcx->ijabx', govvv, t2, r1))
            + asym('0/1|2/3')(einsum('kjmb,imac,kcx->ijabx', gooov, t2, r1))
            - einsum('ijmc,mkab,kcx->ijabx', gooov, t2, r1)
            - einsum('keab,ijec,kcx->ijabx', govvv, t2, r1))
        ar1_cmp = fermitools.math.asym.compound_index(ar1, {0: (0, 1),
                                                            1: (2, 3)})
        return numpy.squeeze(numpy.reshape(ar1_cmp, (ndoubles, cols)))

    return _sigma


def mixed_hessian_left_diff_sigma(fov, gooov, govvv, t2):
    no, _, nv, _ = t2.shape
    noo = no * (no - 1) // 2
    nvv = nv * (nv - 1) // 2
    ndoubles = noo * nvv
    i = mixed_interaction(fov, gooov, govvv)

    def _sigma(r1_flat):
        cols = 1 if r1_flat.ndim is 1 else r1_flat.shape[1]
        r1 = numpy.reshape(r1_flat, (no, nv, cols))
        ar1 = (
            # a terms
            + asym('0/1')(einsum('jcab,icx->ijabx', govvv, r1))
            + asym('2/3')(einsum('ijkb,kax->ijabx', gooov, r1))
            + asym('0/1')(einsum('kcim,mjab,kcx->ijabx', i['o,o'], t2, r1))
            - asym('2/3')(einsum('kcea,ijeb,kcx->ijabx', i['v,v'], t2, r1))
            + asym('0/1|2/3')(einsum('mace,mjeb,icx->ijabx', govvv, t2, r1))
            + asym('0/1|2/3')(einsum('kmie,mjeb,kax->ijabx', gooov, t2, r1))
            + 1./2 * asym('0/1')(einsum('mnjc,mnab,icx->ijabx', gooov, t2, r1))
            + 1./2 * asym('2/3')(einsum('kbef,ijef,kax->ijabx', govvv, t2, r1))
            # b terms
            - asym('0/1')(einsum('kcmi,mjab,kcx->ijabx', i['o,o'], t2, r1))
            + asym('2/3')(einsum('kcae,ijeb,kcx->ijabx', i['v,v'], t2, r1))
            - asym('0/1|2/3')(einsum('jecb,ikae,kcx->ijabx', govvv, t2, r1))
            - asym('0/1|2/3')(einsum('kjmb,imac,kcx->ijabx', gooov, t2, r1))
            + einsum('ijmc,mkab,kcx->ijabx', gooov, t2, r1)
            + einsum('keab,ijec,kcx->ijabx', govvv, t2, r1))
        ar1_cmp = fermitools.math.asym.compound_index(ar1, {0: (0, 1),
                                                            1: (2, 3)})
        return numpy.squeeze(numpy.reshape(ar1_cmp, (ndoubles, cols)))

    return _sigma


def combined_hessian_sigma(dim1, sig_e_orb, sig_e_mix_r, sig_e_mix_l,
                           sig_e_amp):

    def _sigma(z1z2):
        z1, z2 = numpy.split(z1z2, (dim1,))
        return numpy.concatenate((sig_e_orb(z1) + sig_e_mix_r(z2),
                                  sig_e_mix_l(z1) + sig_e_amp(z2)), axis=0)

    return _sigma


def combined_metric_sigma(dim1, sig_s_orb_inv):

    def _sigma(z1z2):
        z1, z2 = numpy.split(z1z2, (dim1,))
        return numpy.concatenate((sig_s_orb_inv(z1), z2), axis=0)

    return _sigma


def main():
    import scripts.ocepa0 as ocepa0

    import numpy
    import scipy.linalg

    import fermitools

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
    ac, bc = interface.hf.unrestricted_orbitals(
            BASIS, LABELS, COORDS, CHARGE, SPIN)
    c_unsrt = scipy.linalg.block_diag(ac, bc)
    sortvec = fermitools.math.spinorb.ab2ov(dim=nbf, na=na, nb=nb)
    c_unsrt = scipy.linalg.block_diag(ac, bc)
    c = fermitools.math.spinorb.sort(c_unsrt, order=sortvec, axes=(1,))

    # Solve OCEPA0
    t2_guess = numpy.zeros((nocc, nocc, norb-nocc, norb-nocc))
    en_elec, c, t2 = ocepa0.solve(
            norb=norb, nocc=nocc, h_aso=h_aso, g_aso=g_aso, c_guess=c,
            t2_guess=t2_guess, niter=200, e_thresh=1e-14, r_thresh=1e-13,
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

    a_orb = orbital_hessian_diag(
            h[o, o], h[v, v], g[o, o, o, o], g[o, o, v, v], g[o, v, o, v],
            g[v, v, v, v], m1[o, o], m1[v, v], m2[o, o, o, o], m2[o, o, v, v],
            m2[o, v, o, v], m2[v, v, v, v])
    a_mix = mixed_hessian_diag(f[o, v], g[o, o, o, v], g[o, v, v, v], t2)
    a_amp = amplitude_hessian(
            f[o, o], f[v, v], g[o, o, o, o], g[o, v, o, v], g[v, v, v, v])

    b_orb = orbital_hessian_offd(
            g[o, o, o, o], g[o, o, v, v], g[o, v, o, v], g[v, v, v, v],
            m2[o, o, o, o], m2[o, o, v, v], m2[o, v, o, v], m2[v, v, v, v])
    b_mix = mixed_hessian_offd(f[o, v], g[o, o, o, v], g[o, v, v, v], t2)
    b_amp = numpy.zeros_like(a_amp)

    # Evaluate dipole polarizability using linear response theory
    p_ao = interface.integrals.dipole(BASIS, LABELS, COORDS)
    p_aso = fermitools.math.spinorb.expand(p_ao, brakets=((1, 2),))
    p = fermitools.math.transform(p_aso, {1: c, 2: c})
    t_orb = orbital_property_gradient(p[:, o, v], m1[o, o], m1[v, v])
    t_amp = amplitude_property_gradient(p[:, o, o], p[:, v, v], t2)

    a = numpy.bmat([[a_orb, a_mix], [a_mix.T, a_amp]])
    b = numpy.bmat([[b_orb, b_mix], [b_mix.T, b_amp]])
    t = numpy.concatenate((t_orb, t_amp), axis=0)
    r = static_response_vector(a, b, t)
    alpha_old = static_linear_response_function(t, r)

    # Evaluate the excitation energies
    s_orb = orbital_metric(m1[o, o], m1[v, v])
    s_amp = numpy.eye(*a_amp.shape)
    s = scipy.linalg.block_diag(s_orb, s_amp)

    e = numpy.bmat([[a, b], [b, a]])
    m = scipy.linalg.block_diag(s, -s)
    w_old = numpy.real(sorted(scipy.linalg.eigvals(e, b=m)))
    print(w_old)

    # Excitation energies
    no = nocc
    nv = norb - nocc
    nsingles = no * nv
    ndoubles = no * (no - 1) * nv * (nv - 1) // 4

    # Orbital terms
    sig_e_orb_sum = orbital_hessian_sum_sigma(
        h[o, o], h[v, v], g[o, o, o, o], g[o, o, v, v], g[o, v, o, v],
        g[v, v, v, v], m1[o, o], m1[v, v], m2[o, o, o, o], m2[o, o, v, v],
        m2[o, v, o, v], m2[v, v, v, v])
    sig_e_orb_diff = orbital_hessian_diff_sigma(
        h[o, o], h[v, v], g[o, o, o, o], g[o, o, v, v], g[o, v, o, v],
        g[v, v, v, v], m1[o, o], m1[v, v], m2[o, o, o, o], m2[o, o, v, v],
        m2[o, v, o, v], m2[v, v, v, v])
    s_orb_inv = scipy.linalg.inv(s_orb)
    sig_s_orb_inv = scipy.sparse.linalg.aslinearoperator(s_orb_inv)
    # Mixted right terms
    rsig_e_mix_sum = mixed_hessian_right_sum_sigma(
        f[o, v], g[o, o, o, v], g[o, v, v, v], t2)
    rsig_e_mix_diff = mixed_hessian_right_diff_sigma(
        f[o, v], g[o, o, o, v], g[o, v, v, v], t2)
    # Mixed left terms
    lsig_e_mix_sum = mixed_hessian_left_sum_sigma(
        f[o, v], g[o, o, o, v], g[o, v, v, v], t2)
    lsig_e_mix_diff = mixed_hessian_left_diff_sigma(
        f[o, v], g[o, o, o, v], g[o, v, v, v], t2)
    # Amplitude terms
    sig_e_amp_sum = sig_e_amp_diff = amplitude_hessian_sigma(
        f[o, o], f[v, v], g[o, o, o, o], g[o, v, o, v], g[v, v, v, v])

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
    print(alpha_old.round(10))
    print(numpy.diag(alpha) / numpy.diag(alpha_old))


if __name__ == '__main__':
    main()

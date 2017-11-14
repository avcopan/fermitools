import numpy

import fermitools
from fermitools.math import einsum
from fermitools.math.asym import antisymmetrizer_product as asym

import interfaces.psi4 as interface
from .lr_scf import diagonal_orbital_hessian
from .lr_scf import offdiagonal_orbital_hessian
from .lr_scf import orbital_property_gradient
from .lr_scf import diagonal_orbital_metric
from .lr_scf import static_response_vector
from .lr_scf import static_linear_response_function
from .lr_scf import solve_static_response_vector

# sigmas
from .lr_scf import diagonal_orbital_hessian_sigma
from .lr_scf import offdiagonal_orbital_hessian_sigma
from .lr_scf import diagonal_orbital_metric_sigma
from .lr_scf import orbital_hessian_sum_sigma
from .lr_scf import orbital_hessian_diff_sigma


def diagonal_amplitude_hessian(foo, fvv, goooo, govov, gvvvv):
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


def diagonal_mixed_hessian(fov, gooov, govvv, t2):
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


def offdiagonal_mixed_hessian(fov, gooov, govvv, t2):
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


def diagonal_amplitude_hessian_sigma(foo, fvv, goooo, govov, gvvvv):
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


def diagonal_mixed_hessian_right_sigma(fov, gooov, govvv, t2):
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
        ar2 = (+ 1./2 * einsum('lacd,ilcdx->iax', govvv, r2)
               + 1./2 * einsum('klid,kladx->iax', gooov, r2)
               + 1./2 * einsum('iakm,mlcd,klcdx->iax', i['o,o'], t2, r2)
               - 1./2 * einsum('iaec,kled,klcdx->iax', i['v,v'], t2, r2)
               + einsum('mcae,mled,ilcdx->iax', govvv, t2, r2)
               + einsum('imke,mled,kladx->iax', gooov, t2, r2)
               + 1./4 * einsum('mnla,mncd,ilcdx->iax', gooov, t2, r2)
               + 1./4 * einsum('idef,klef,kladx->iax', govvv, t2, r2))
        return numpy.squeeze(numpy.reshape(ar2, (nsingles, cols)))

    return _sigma


def offdiagonal_mixed_hessian_right_sigma(fov, gooov, govvv, t2):
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
        ar2 = (+ 1./2 * einsum('iamk,mlcd,klcdx->iax', i['o,o'], t2, r2)
               - 1./2 * einsum('iace,kled,klcdx->iax', i['v,v'], t2, r2)
               + einsum('lead,kice,klcdx->iax', govvv, t2, r2)
               + einsum('ilmd,kmca,klcdx->iax', gooov, t2, r2)
               - 1./4 * einsum('klma,micd,klcdx->iax', gooov, t2, r2)
               - 1./4 * einsum('iecd,klea,klcdx->iax', govvv, t2, r2))
        return numpy.squeeze(numpy.reshape(ar2, (nsingles, cols)))

    return _sigma


def diagonal_mixed_hessian_left_sigma(fov, gooov, govvv, t2):
    no, _, nv, _ = t2.shape
    noo = no * (no - 1) // 2
    nvv = nv * (nv - 1) // 2
    ndoubles = noo * nvv
    i = mixed_interaction(fov, gooov, govvv)

    def _sigma(r1_flat):
        cols = 1 if r1_flat.ndim is 1 else r1_flat.shape[1]
        r1 = numpy.reshape(r1_flat, (no, nv, cols))
        ar1 = (+ asym('0/1')(einsum('jcab,icx->ijabx', govvv, r1))
               + asym('2/3')(einsum('ijkb,kax->ijabx', gooov, r1))
               + asym('0/1')(einsum('kcim,mjab,kcx->ijabx', i['o,o'], t2, r1))
               - asym('2/3')(einsum('kcea,ijeb,kcx->ijabx', i['v,v'], t2, r1))
               + asym('0/1|2/3')(einsum('mace,mjeb,icx->ijabx', govvv, t2, r1))
               + asym('0/1|2/3')(einsum('kmie,mjeb,kax->ijabx', gooov, t2, r1))
               + 1./2
               * asym('0/1')(einsum('mnjc,mnab,icx->ijabx', gooov, t2, r1))
               + 1./2
               * asym('2/3')(einsum('kbef,ijef,kax->ijabx', govvv, t2, r1)))
        ar1_cmp = fermitools.math.asym.compound_index(ar1, {0: (0, 1),
                                                            1: (2, 3)})
        return numpy.squeeze(numpy.reshape(ar1_cmp, (ndoubles, cols)))

    return _sigma


def offdiagonal_mixed_hessian_left_sigma(fov, gooov, govvv, t2):
    no, _, nv, _ = t2.shape
    noo = no * (no - 1) // 2
    nvv = nv * (nv - 1) // 2
    ndoubles = noo * nvv
    i = mixed_interaction(fov, gooov, govvv)

    def _sigma(r1_flat):
        cols = 1 if r1_flat.ndim is 1 else r1_flat.shape[1]
        r1 = numpy.reshape(r1_flat, (no, nv, cols))
        ar1 = (+ asym('0/1')(einsum('kcmi,mjab,kcx->ijabx', i['o,o'], t2, r1))
               - asym('2/3')(einsum('kcae,ijeb,kcx->ijabx', i['v,v'], t2, r1))
               + asym('0/1|2/3')(einsum('jecb,ikae,kcx->ijabx', govvv, t2, r1))
               + asym('0/1|2/3')(einsum('kjmb,imac,kcx->ijabx', gooov, t2, r1))
               - einsum('ijmc,mkab,kcx->ijabx', gooov, t2, r1)
               - einsum('keab,ijec,kcx->ijabx', govvv, t2, r1))
        ar1_cmp = fermitools.math.asym.compound_index(ar1, {0: (0, 1),
                                                            1: (2, 3)})
        return numpy.squeeze(numpy.reshape(ar1_cmp, (ndoubles, cols)))

    return _sigma


# Sum/difference sigmas
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
    ac, bc = interface.hf.unrestricted_orbitals(BASIS, LABELS, COORDS,
                                                CHARGE, SPIN)
    c_unsrt = scipy.linalg.block_diag(ac, bc)
    sortvec = fermitools.math.spinorb.ab2ov(dim=nbf, na=na, nb=nb)
    c_unsrt = scipy.linalg.block_diag(ac, bc)
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

    a_orb = diagonal_orbital_hessian(h[o, o], h[v, v], g[o, o, o, o],
                                     g[o, o, v, v], g[o, v, o, v],
                                     g[v, v, v, v], m1[o, o], m1[v, v],
                                     m2[o, o, o, o], m2[o, o, v, v],
                                     m2[o, v, o, v], m2[v, v, v, v])
    a_mix = diagonal_mixed_hessian(f[o, v], g[o, o, o, v], g[o, v, v, v], t2)
    a_amp = diagonal_amplitude_hessian(f[o, o], f[v, v], g[o, o, o, o],
                                       g[o, v, o, v], g[v, v, v, v])

    b_orb = offdiagonal_orbital_hessian(g[o, o, o, o], g[o, o, v, v],
                                        g[o, v, o, v], g[v, v, v, v],
                                        m2[o, o, o, o], m2[o, o, v, v],
                                        m2[o, v, o, v], m2[v, v, v, v])
    b_mix = offdiagonal_mixed_hessian(f[o, v], g[o, o, o, v], g[o, v, v, v],
                                      t2)
    b_amp = numpy.zeros_like(a_amp)

    # Test the orbital and amplitude Hessians
    import os
    from numpy.testing import assert_almost_equal

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'data')
    en_dxdx = numpy.load(os.path.join(data_path,
                                      'lr_ocepa0/neutral/en_dxdx.npy'))
    en_dtdx = numpy.load(os.path.join(data_path,
                                      'lr_ocepa0/neutral/en_dtdx.npy'))
    en_dxdt = numpy.load(os.path.join(data_path,
                                      'lr_ocepa0/neutral/en_dxdt.npy'))
    en_dtdt = numpy.load(os.path.join(data_path,
                                      'lr_ocepa0/neutral/en_dtdt.npy'))

    assert_almost_equal(en_dxdx, 2*(a_orb + b_orb), decimal=9)
    assert_almost_equal(en_dxdt, 2*(a_mix + b_mix), decimal=9)
    assert_almost_equal(en_dxdt, numpy.transpose(en_dtdx), decimal=9)
    assert_almost_equal(en_dtdt, 2*(a_amp + b_amp), decimal=9)

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
    s_orb = diagonal_orbital_metric(m1[o, o], m1[v, v])
    s_amp = numpy.eye(*a_amp.shape)
    s = scipy.linalg.block_diag(s_orb, s_amp)

    e = numpy.bmat([[a, b], [b, a]])
    m = scipy.linalg.block_diag(s, -s)
    w_old = numpy.real(sorted(scipy.linalg.eigvals(e, b=m)))
    print(w_old)

    # Test sigmas
    no = nocc
    nv = norb - nocc
    nsingles = no * nv
    ndoubles = no * (no - 1) * nv * (nv - 1) // 4
    i1 = numpy.eye(nsingles)
    i2 = numpy.eye(ndoubles)
    sig_a_orb = diagonal_orbital_hessian_sigma(
            h[o, o], h[v, v], g[o, o, o, o], g[o, o, v, v], g[o, v, o, v],
            g[v, v, v, v], m1[o, o], m1[v, v], m2[o, o, o, o], m2[o, o, v, v],
            m2[o, v, o, v], m2[v, v, v, v])
    sig_b_orb = offdiagonal_orbital_hessian_sigma(
            g[o, o, o, o], g[o, o, v, v], g[o, v, o, v], g[v, v, v, v],
            m2[o, o, o, o], m2[o, o, v, v], m2[o, v, o, v], m2[v, v, v, v])
    sig_s_orb = diagonal_orbital_metric_sigma(m1[o, o], m1[v, v])
    rsig_a_mix = diagonal_mixed_hessian_right_sigma(
            f[o, v], g[o, o, o, v], g[o, v, v, v], t2)
    rsig_b_mix = offdiagonal_mixed_hessian_right_sigma(
            f[o, v], g[o, o, o, v], g[o, v, v, v], t2)
    lsig_a_mix = diagonal_mixed_hessian_left_sigma(
            f[o, v], g[o, o, o, v], g[o, v, v, v], t2)
    lsig_b_mix = offdiagonal_mixed_hessian_left_sigma(
            f[o, v], g[o, o, o, v], g[o, v, v, v], t2)
    sig_a_amp = diagonal_amplitude_hessian_sigma(
            f[o, o], f[v, v], g[o, o, o, o], g[o, v, o, v], g[v, v, v, v])

    def sig_b_amp(r2_flat):
        return numpy.zeros_like(r2_flat)

    def sig_s_amp(r2_flat):
        return r2_flat

    def sig_a(r1r2):
        r1, r2 = numpy.split(r1r2, (nsingles,))
        return numpy.concatenate((sig_a_orb(r1) + rsig_a_mix(r2),
                                  lsig_a_mix(r1) + sig_a_amp(r2)), axis=0)

    def sig_b(r1r2):
        r1, r2 = numpy.split(r1r2, (nsingles,))
        return numpy.concatenate((sig_b_orb(r1) + rsig_b_mix(r2),
                                  lsig_b_mix(r1) + sig_b_amp(r2)), axis=0)

    def sig_s(r1r2):
        r1, r2 = numpy.split(r1r2, (nsingles,))
        return numpy.concatenate((sig_s_orb(r1), sig_s_amp(r2)), axis=0)

    print(numpy.linalg.norm(sig_a_orb(i1) - a_orb))
    print(numpy.linalg.norm(sig_b_orb(i1) - b_orb))
    print(numpy.linalg.norm(sig_s_orb(i1) - s_orb))
    print(numpy.linalg.norm(rsig_a_mix(i2) - a_mix))
    print(numpy.linalg.norm(rsig_b_mix(i2) - b_mix))
    print(numpy.linalg.norm(lsig_a_mix(i1) - numpy.transpose(a_mix)))
    print(numpy.linalg.norm(lsig_b_mix(i1) - numpy.transpose(b_mix)))
    print(numpy.linalg.norm(sig_a_amp(i2) - a_amp))
    print(numpy.linalg.norm(sig_b_amp(i2) - b_amp))
    print(numpy.linalg.norm(sig_s_amp(i2) - s_amp))

    # Excitation energies
    # Orbital terms
    sig_e_orb_plus = orbital_hessian_sum_sigma(
        h[o, o], h[v, v], g[o, o, o, o], g[o, o, v, v], g[o, v, o, v],
        g[v, v, v, v], m1[o, o], m1[v, v], m2[o, o, o, o], m2[o, o, v, v],
        m2[o, v, o, v], m2[v, v, v, v])
    sig_e_orb_minus = orbital_hessian_diff_sigma(
        h[o, o], h[v, v], g[o, o, o, o], g[o, o, v, v], g[o, v, o, v],
        g[v, v, v, v], m1[o, o], m1[v, v], m2[o, o, o, o], m2[o, o, v, v],
        m2[o, v, o, v], m2[v, v, v, v])
    s_orb_inv = scipy.linalg.inv(s_orb)
    sig_s_orb_inv = scipy.sparse.linalg.aslinearoperator(s_orb_inv)
    # Mixted right terms
    rsig_e_mix_plus = mixed_hessian_right_sum_sigma(
        f[o, v], g[o, o, o, v], g[o, v, v, v], t2)
    rsig_e_mix_minus = mixed_hessian_right_diff_sigma(
        f[o, v], g[o, o, o, v], g[o, v, v, v], t2)
    # Mixed left terms
    lsig_e_mix_plus = mixed_hessian_left_sum_sigma(
        f[o, v], g[o, o, o, v], g[o, v, v, v], t2)
    lsig_e_mix_minus = mixed_hessian_left_diff_sigma(
        f[o, v], g[o, o, o, v], g[o, v, v, v], t2)
    # Amplitude terms
    sig_e_amp_plus = sig_e_amp_minus = diagonal_amplitude_hessian_sigma(
        f[o, o], f[v, v], g[o, o, o, o], g[o, v, o, v], g[v, v, v, v])
    sig_s_amp_inv = sig_s_amp

    def sig_e_plus(z1z2):
        z1, z2 = numpy.split(z1z2, (nsingles,))
        return numpy.concatenate((sig_e_orb_plus(z1) + rsig_e_mix_plus(z2),
                                  lsig_e_mix_plus(z1) + sig_e_amp_plus(z2)),
                                 axis=0)

    def sig_e_minus(z1z2):
        z1, z2 = numpy.split(z1z2, (nsingles,))
        return numpy.concatenate((sig_e_orb_minus(z1) + rsig_e_mix_minus(z2),
                                  lsig_e_mix_minus(z1) + sig_e_amp_minus(z2)),
                                 axis=0)

    def sig_s_inv(z1z2):
        z1, z2 = numpy.split(z1z2, (nsingles,))
        return numpy.concatenate((sig_s_orb_inv(z1), sig_s_amp_inv(z2)),
                                 axis=0)

    def sig_e_eff(z1z2):
        return sig_s_inv(sig_e_plus(sig_s_inv(sig_e_minus(z1z2))))

    dim = nsingles + ndoubles
    e_ = scipy.sparse.linalg.LinearOperator((dim, dim), matvec=sig_e_eff)
    w2, u = scipy.sparse.linalg.eigs(e_, k=dim-2, which='SR')
    w = numpy.sqrt(numpy.real(sorted(w2)))
    print(w)
    print(w / w_old[dim:2*dim-2])

    # Response function
    r = solve_static_response_vector(dim, sig_a, sig_b, t)
    alpha = numpy.tensordot(r, t, axes=(0, 0))
    print(alpha.round(10))
    print(alpha_old.round(10))
    print(numpy.diag(alpha) / numpy.diag(alpha_old))


if __name__ == '__main__':
    main()

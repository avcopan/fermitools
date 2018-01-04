import numpy
import scipy
from toolz import functoolz

from ..math import einsum
from ..math import transform
from ..math import broadcast_sum
from ..math import raveler, unraveler
from ..math.asym import antisymmetrizer_product as asm
from ..math.asym import megaraveler, megaunraveler
from ..math.linalg.direct import eye
from ..math.linalg.direct import negative
from ..math.linalg.direct import bmat
from ..math.linalg.direct import block_diag
from ..math.linalg.direct import evec_guess
from ..math.linalg.direct import eighg

from ..oo.odc12 import fock_xy
from ..oo.odc12 import fancy_property
from ..oo.odc12 import onebody_density


def solve_static_response(a, b, pg):
    a_ = a(numpy.eye(len(pg)))
    b_ = b(numpy.eye(len(pg)))

    e = a_ + b_
    r = scipy.linalg.solve(e, -2*pg)
    return r


def solve_spectrum(a, b, s, d, ad, sd, nroot=1, nvec=None, niter=50,
                   r_thresh=1e-6):
    nvec = 2 * nroot if nvec is None else nvec

    e = bmat([[a, b], [b, a]], 2)
    m = bmat([[s, d], [negative(d), negative(s)]], 2)
    ed = numpy.concatenate((+ad, +ad))
    md = numpy.concatenate((+sd, -sd))

    guess = evec_guess(md, nvec, bd=ed)
    v, u, info = eighg(
            a=m, b=e, neig=nroot, ad=md, bd=ed, guess=guess,
            r_thresh=r_thresh, nvec=nvec, niter=niter)
    w = -1. / v

    return w, u, info


def hessian_sigma(hoo, hov, hvv, goooo, gooov, goovv, govov, govvv, gvvvv, t2,
                  complex=False):
    no, _, nv, _ = t2.shape
    n1 = no * nv
    r1 = raveler({0: (0, 1)})
    u1 = unraveler({0: {0: no, 1: nv}})
    r2 = megaraveler({0: ((0, 1), (2, 3))})
    u2 = megaunraveler({0: {(0, 1): no, (2, 3): nv}})

    m1oo, m1vv = onebody_density(t2)
    foo = fock_xy(hxy=hoo, goxoy=goooo, gxvyv=govov, m1oo=m1oo, m1vv=m1vv)
    fov = fock_xy(hxy=hov, goxoy=gooov, gxvyv=govvv, m1oo=m1oo, m1vv=m1vv)
    fvv = fock_xy(hxy=hvv, goxoy=govov, gxvyv=gvvvv, m1oo=m1oo, m1vv=m1vv)
    ffoo = fancy_property(foo, m1oo)
    ffvv = fancy_property(fvv, m1vv)
    fioo, fivv = fancy_mixed_interaction(fov, gooov, govvv, m1oo, m1vv)
    fgoooo, fgovov, fgvvvv = fancy_repulsion(
            ffoo, ffvv, goooo, govov, gvvvv, m1oo, m1vv)

    a11u = a11_sigma(foo, fvv, goooo, goovv, govov, gvvvv, m1oo, m1vv, t2)
    a12u = a12_sigma(gooov, govvv, fioo, fivv, t2)
    a21u = a21_sigma(gooov, govvv, fioo, fivv, t2)
    a22u = a22_sigma(ffoo, ffvv, goooo, govov, gvvvv, fgoooo, fgovov, fgvvvv,
                     t2)
    a11 = functoolz.compose(r1, a11u, u1)
    a12 = functoolz.compose(r1, a12u, u2)
    a21 = functoolz.compose(r2, a21u, u1)
    a22 = functoolz.compose(r2, a22u, u2)
    a = bmat([[a11, a12], [a21, a22]], (n1,))

    if complex:
        b11u = b11_sigma(goooo, goovv, govov, gvvvv, m1oo, m1vv, t2)
        b12u = b12_sigma(gooov, govvv, fioo, fivv, t2)
        b21u = b21_sigma(gooov, govvv, fioo, fivv, t2)
        b22u = b22_sigma(fgoooo, fgovov, fgvvvv, t2)
        b11 = functoolz.compose(r1, b11u, u1)
        b12 = functoolz.compose(r1, b12u, u2)
        b21 = functoolz.compose(r2, b21u, u1)
        b22 = functoolz.compose(r2, b22u, u2)
        b = bmat([[b11, b12], [b21, b22]], (n1,))
        return a, b
    else:
        return a


def metric_sigma(t2):
    no, _, nv, _ = t2.shape
    n1 = no * nv
    r1 = raveler({0: (0, 1)})
    u1 = unraveler({0: {0: no, 1: nv}})

    m1oo, m1vv = onebody_density(t2)

    s11u = s11_sigma(m1oo, m1vv)
    s11 = functoolz.compose(r1, s11u, u1)
    s = block_diag((s11, eye), (n1,))
    return s


def approximate_diagonal_hessian(hoo, hvv, goooo, govov, gvvvv, t2):
    no, _, nv, _ = t2.shape
    r1 = raveler({0: (0, 1)})
    r2 = megaraveler({0: ((0, 1), (2, 3))})

    m1oo, m1vv = onebody_density(t2)
    foo = fock_xy(hxy=hoo, goxoy=goooo, gxvyv=govov, m1oo=m1oo, m1vv=m1vv)
    fvv = fock_xy(hxy=hvv, goxoy=govov, gxvyv=gvvvv, m1oo=m1oo, m1vv=m1vv)
    ffoo = fancy_property(foo, m1oo)
    ffvv = fancy_property(fvv, m1vv)

    eo = numpy.diagonal(foo)
    ev = numpy.diagonal(fvv)
    efo = numpy.diagonal(ffoo)
    efv = numpy.diagonal(ffvv)

    ad1 = r1(broadcast_sum({0: -eo, 1: +ev}))
    ad2 = r2(broadcast_sum({0: -efo, 1: -efo, 2: -efv, 3: -efv}))
    return numpy.concatenate((ad1, ad2), axis=0)


def property_gradient(poo, pov, pvv, t2):
    r1 = raveler({0: (0, 1)})
    r2 = megaraveler({0: ((0, 1), (2, 3))})

    m1oo, m1vv = onebody_density(t2)
    fpoo = fancy_property(poo, m1oo)
    fpvv = fancy_property(pvv, m1vv)
    pg1 = r1(onebody_property_gradient(pov, m1oo, m1vv))
    pg2 = r2(twobody_property_gradient(fpoo, -fpvv, t2))
    return numpy.concatenate((pg1, pg2), axis=0)


def onebody_property_gradient(pov, m1oo, m1vv):
    return (
        + einsum('...ie,ea->ia...', pov, m1vv)
        - einsum('im,...ma->ia...', m1oo, pov))


def twobody_property_gradient(poo, pvv, t2):
    return (
        + asm('2/3')(
              einsum('...ac,ijcb->ijab...', pvv, t2))
        - asm('0/1')(
              einsum('...ik,kjab->ijab...', poo, t2)))


def a11_sigma(foo, fvv, goooo, goovv, govov, gvvvv, m1oo, m1vv, t2):
    fcoo = (numpy.dot(foo, m1oo)
            + 1./2 * einsum('imef,jmef->ij', goovv, t2)
            + 1./4 * einsum('imno,jmcd,nocd->ij', goooo, t2, t2)
            - einsum('iemf,mkec,jkfc->ij', govov, t2, t2))
    fcvv = (numpy.dot(fvv, m1vv)
            + 1./2 * einsum('mnae,mnbe->ab', goovv, t2)
            + 1./4 * einsum('aefg,klbe,klfg->ab', gvvvv, t2, t2)
            - einsum('nema,mkec,nkbc->ab', govov, t2, t2))
    fsoo = (fcoo + numpy.transpose(fcoo)) / 2.
    fsvv = (fcvv + numpy.transpose(fcvv)) / 2.

    def _a11(r1):
        return (
            - einsum('ab,ib...->ia...', fsvv, r1)
            - einsum('ij,ja...->ia...', fsoo, r1)
            + einsum('ij,ab,jb...->ia...', foo, m1vv, r1)
            + einsum('ab,ij,jb...->ia...', fvv, m1oo, r1)
            - einsum('manb,mj,in,jb...->ia...', govov, m1oo, m1oo, r1)
            - einsum('iejf,af,eb,jb...->ia...', govov, m1vv, m1vv, r1)
            + einsum('jame,im,be,jb...->ia...', govov, m1oo, m1vv, r1)
            + einsum('ibme,jm,ae,jb...->ia...', govov, m1oo, m1vv, r1)
            - einsum('minj,nkac,mkbc,jb...->ia...', goooo, t2, t2, r1)
            + 1./2 * einsum('manb,micd,njcd,jb...->ia...', govov, t2, t2, r1)
            + 1./2 * einsum('iejf,klae,klbf,jb...->ia...', govov, t2, t2, r1)
            - einsum('aebf,jkec,ikfc,jb...->ia...', gvvvv, t2, t2, r1)
            - einsum('ibme,mkac,jkec,jb...->ia...', govov, t2, t2, r1)
            - einsum('jame,mkbc,ikec,jb...->ia...', govov, t2, t2, r1)
            )

    return _a11


def b11_sigma(goooo, goovv, govov, gvvvv, m1oo, m1vv, t2):

    def _b11(r1):
        return (
            + einsum('jema,imbe,jb...->ia...', govov, t2, r1)
            + einsum('iemb,jmae,jb...->ia...', govov, t2, r1)
            + 1./2 * einsum('ijmn,mnab,jb...->ia...', goooo, t2, r1)
            + 1./2 * einsum('efab,ijef,jb...->ia...', gvvvv, t2, r1)
            + einsum('imbe,jm,ea,jb...->ia...', goovv, m1oo, m1vv, r1)
            + einsum('jmae,im,eb,jb...->ia...', goovv, m1oo, m1vv, r1)
            + einsum('mnab,im,jn,jb...->ia...', goovv, m1oo, m1oo, r1)
            + einsum('ijef,ea,fb,jb...->ia...', goovv, m1vv, m1vv, r1)
            + 1./4 * einsum('mnab,ijcd,mncd,jb...->ia...', goovv, t2, t2, r1)
            - einsum('imbe,mkec,jkac,jb...->ia...', goovv, t2, t2, r1)
            - einsum('jmae,mkec,ikbc,jb...->ia...', goovv, t2, t2, r1)
            + 1./4 * einsum('ijef,klef,klab,jb...->ia...', goovv, t2, t2, r1)
            )

    return _b11


def s11_sigma(m1oo, m1vv):

    def _s11(r1):
        return (
            + einsum('ij,ja...->ia...', m1oo, r1)
            - einsum('ab,ib...->ia...', m1vv, r1))

    return _s11


def fancy_mixed_interaction(fov, gooov, govvv, m1oo, m1vv):
    no, uo = scipy.linalg.eigh(m1oo)
    nv, uv = scipy.linalg.eigh(m1vv)
    n1oo = broadcast_sum({2: no, 3: no}) - 1
    n1vv = broadcast_sum({2: nv, 3: nv}) - 1
    io = numpy.eye(*uo.shape)
    iv = numpy.eye(*uv.shape)
    ioo = (+ einsum('ik,la->iakl', io, fov)
           - einsum('mlka,im->iakl', gooov, m1oo)
           + einsum('ilke,ae->iakl', gooov, m1vv))
    ivv = (- einsum('ac,id->iadc', iv, fov)
           + einsum('mcad,im->iadc', govvv, m1oo)
           - einsum('iced,ae->iadc', govvv, m1vv))
    tfioo = transform(ioo, {2: uo, 3: uo}) / n1oo
    tfivv = transform(ivv, {2: uv, 3: uv}) / n1vv
    fioo = transform(tfioo, {2: uo.T, 3: uo.T})
    fivv = transform(tfivv, {2: uv.T, 3: uv.T})
    return fioo, fivv


def fancy_repulsion(ffoo, ffvv, goooo, govov, gvvvv, m1oo, m1vv):
    no, uo = scipy.linalg.eigh(m1oo)
    nv, uv = scipy.linalg.eigh(m1vv)
    n1oo = broadcast_sum({0: no, 1: no}) - 1
    n1vv = broadcast_sum({0: nv, 1: nv}) - 1
    io = numpy.eye(*uo.shape)
    iv = numpy.eye(*uv.shape)
    tffoo = transform(ffoo, {0: uo, 1: uo})
    tffvv = transform(ffvv, {0: uv, 1: uv})
    tgoooo = transform(goooo, {0: uo, 1: uo, 2: uo, 3: uo})
    tgovov = transform(govov, {0: uo, 1: uv, 2: uo, 3: uv})
    tgvvvv = transform(gvvvv, {0: uv, 1: uv, 2: uv, 3: uv})
    tfgoooo = ((tgoooo - einsum('il,jk->ikjl', tffoo, io)
                       - einsum('il,jk->ikjl', io, tffoo))
               / einsum('ij,kl->ikjl', n1oo, n1oo))
    tfgovov = tgovov / einsum('ij,ab->iajb', n1oo, n1vv)
    tfgvvvv = ((tgvvvv - einsum('ad,bc->acbd', tffvv, iv)
                       - einsum('ad,bc->acbd', iv, tffvv))
               / einsum('ab,cd->acbd', n1vv, n1vv))
    fgoooo = transform(tfgoooo, {0: uo.T, 1: uo.T, 2: uo.T, 3: uo.T})
    fgovov = transform(tfgovov, {0: uo.T, 1: uv.T, 2: uo.T, 3: uv.T})
    fgvvvv = transform(tfgvvvv, {0: uv.T, 1: uv.T, 2: uv.T, 3: uv.T})
    return fgoooo, fgovov, fgvvvv


def a12_sigma(gooov, govvv, fioo, fivv, t2):

    def _a12(r2):
        return (
            + 1./2 * einsum('lacd,ilcd...->ia...', govvv, r2)
            + 1./2 * einsum('klid,klad...->ia...', gooov, r2)
            + 1./2 * einsum('iakm,mlcd,klcd...->ia...', fioo, t2, r2)
            + 1./2 * einsum('iaec,kled,klcd...->ia...', fivv, t2, r2)
            + einsum('mcae,mled,ilcd...->ia...', govvv, t2, r2)
            + einsum('imke,mled,klad...->ia...', gooov, t2, r2)
            + 1./4 * einsum('mnla,mncd,ilcd...->ia...', gooov, t2, r2)
            + 1./4 * einsum('idef,klef,klad...->ia...', govvv, t2, r2))

    return _a12


def b12_sigma(gooov, govvv, fioo, fivv, t2):

    def _b12(r2):
        return (
            + 1./2 * einsum('iamk,mlcd,klcd...->ia...', fioo, t2, r2)
            + 1./2 * einsum('iace,kled,klcd...->ia...', fivv, t2, r2)
            + einsum('lead,kice,klcd...->ia...', govvv, t2, r2)
            + einsum('ilmd,kmca,klcd...->ia...', gooov, t2, r2)
            - 1./4 * einsum('klma,micd,klcd...->ia...', gooov, t2, r2)
            - 1./4 * einsum('iecd,klea,klcd...->ia...', govvv, t2, r2))

    return _b12


def a21_sigma(gooov, govvv, fioo, fivv, t2):

    def _a21(r1):
        return (
            + asm('0/1')(
                   einsum('jcab,ic...->ijab...', govvv, r1))
            + asm('2/3')(
                   einsum('ijkb,ka...->ijab...', gooov, r1))
            + asm('0/1')(
                   einsum('kcim,mjab,kc...->ijab...', fioo, t2, r1))
            + asm('2/3')(
                   einsum('kcea,ijeb,kc...->ijab...', fivv, t2, r1))
            + asm('0/1|2/3')(
                   einsum('mace,mjeb,ic...->ijab...', govvv, t2, r1))
            + asm('0/1|2/3')(
                   einsum('kmie,mjeb,ka...->ijab...', gooov, t2, r1))
            + 1./2 * asm('0/1')(
                   einsum('mnjc,mnab,ic...->ijab...', gooov, t2, r1))
            + 1./2 * asm('2/3')(
                   einsum('kbef,ijef,ka...->ijab...', govvv, t2, r1)))

    return _a21


def b21_sigma(gooov, govvv, fioo, fivv, t2):

    def _b21(r1):
        return (
            + asm('0/1')(
                   einsum('kcmi,mjab,kc...->ijab...', fioo, t2, r1))
            + asm('2/3')(
                   einsum('kcae,ijeb,kc...->ijab...', fivv, t2, r1))
            + asm('0/1|2/3')(
                   einsum('jecb,ikae,kc...->ijab...', govvv, t2, r1))
            + asm('0/1|2/3')(
                   einsum('kjmb,imac,kc...->ijab...', gooov, t2, r1))
            - einsum('ijmc,mkab,kc...->ijab...', gooov, t2, r1)
            - einsum('keab,ijec,kc...->ijab...', govvv, t2, r1))

    return _b21


def a22_sigma(ffoo, ffvv, goooo, govov, gvvvv, fgoooo, fgovov, fgvvvv, t2):

    def _a22(r2):
        return (
            - asm('2/3')(einsum('ac,ijcb...->ijab...', ffvv, r2))
            - asm('0/1')(einsum('ik,kjab...->ijab...', ffoo, r2))
            + 1./2 * einsum('abcd,ijcd...->ijab...', gvvvv, r2)
            + 1./2 * einsum('ijkl,klab...->ijab...', goooo, r2)
            - asm('0/1|2/3')(einsum('jcla,ilcb...->ijab...', govov, r2))
            + 1./2 * asm('2/3')(
                einsum('afec,ijeb,klfd,klcd...->ijab...', fgvvvv, t2, t2, r2))
            + 1./2 * asm('2/3')(
                einsum('kame,ijeb,mlcd,klcd...->ijab...', fgovov, t2, t2, r2))
            + 1./2 * asm('0/1')(
                einsum('meic,mjab,kled,klcd...->ijab...', fgovov, t2, t2, r2))
            + 1./2 * asm('0/1')(
                einsum('mkin,mjab,nlcd,klcd...->ijab...', fgoooo, t2, t2, r2)))

    return _a22


def b22_sigma(fgoooo, fgovov, fgvvvv, t2):

    def _b22(r2):
        return (
            + 1./2 * asm('2/3')(
                einsum('acef,ijeb,klfd,klcd...->ijab...', fgvvvv, t2, t2, r2))
            + 1./2 * asm('2/3')(
                einsum('nake,ijeb,nlcd,klcd...->ijab...', fgovov, t2, t2, r2))
            + 1./2 * asm('0/1')(
                einsum('mcif,mjab,klfd,klcd...->ijab...', fgovov, t2, t2, r2))
            + 1./2 * asm('0/1')(
                einsum('mnik,mjab,nlcd,klcd...->ijab...', fgoooo, t2, t2, r2)))

    return _b22

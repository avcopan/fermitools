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
from ..math.linalg.direct import zero
from ..math.linalg.direct import bmat
from ..math.linalg.direct import block_diag

from ..oo.odc12 import fancy_property
from ..oo.odc12 import onebody_density


# Public
def hessian_zeroth_order_diagonal(foo, fvv, t2):
    r1 = raveler({0: (0, 1)})
    r2 = megaraveler({0: ((0, 1), (2, 3))})

    ad1u = onebody_hessian_zeroth_order_diagonal(foo, fvv)
    ad2u = twobody_hessian_zeroth_order_diagonal(foo, fvv, t2)
    ad1 = r1(ad1u)
    ad2 = r2(ad2u)
    return numpy.concatenate((ad1, ad2), axis=0)


def metric_zeroth_order_diagonal(no, nv):
    n1 = no * nv
    n2 = no * (no - 1) * nv * (nv - 1) // 4
    return numpy.ones(n1+n2)


def property_gradient(poo, pov, pvv, t2):
    r1 = raveler({0: (0, 1)})
    r2 = megaraveler({0: ((0, 1), (2, 3))})

    m1oo, m1vv = onebody_density(t2)
    fpoo = fancy_property(poo, m1oo)
    fpvv = fancy_property(pvv, m1vv)
    pg1 = r1(onebody_property_gradient(pov, m1oo, m1vv))
    pg2 = r2(twobody_property_gradient(fpoo, -fpvv, t2))
    return numpy.concatenate((pg1, pg2), axis=0)


def hessian(foo, fov, fvv, goooo, gooov, goovv, govov, govvv, gvvvv, t2):
    no, _, nv, _ = t2.shape
    n1 = no * nv
    r1 = raveler({0: (0, 1)})
    u1 = unraveler({0: {0: no, 1: nv}})
    r2 = megaraveler({0: ((0, 1), (2, 3))})
    u2 = megaunraveler({0: {(0, 1): no, (2, 3): nv}})

    a11u, b11u = onebody_hessian(foo, fvv, goooo, goovv, govov, gvvvv, t2)
    a12u, b12u = mixed_upper_hessian(fov, gooov, govvv, t2)
    a21u, b21u = mixed_lower_hessian(fov, gooov, govvv, t2)
    a22u, b22u = twobody_hessian(foo, fvv, goooo, govov, gvvvv, t2)
    a11 = functoolz.compose(r1, a11u, u1)
    b11 = functoolz.compose(r1, b11u, u1)
    a12 = functoolz.compose(r1, a12u, u2)
    b12 = functoolz.compose(r1, b12u, u2)
    a21 = functoolz.compose(r2, a21u, u1)
    b21 = functoolz.compose(r2, b21u, u1)
    a22 = functoolz.compose(r2, a22u, u2)
    b22 = functoolz.compose(r2, b22u, u2)
    a = bmat([[a11, a12], [a21, a22]], (n1,))
    b = bmat([[b11, b12], [b21, b22]], (n1,))
    return a, b


def metric(t2):
    no, _, nv, _ = t2.shape
    n1 = no * nv
    r1 = raveler({0: (0, 1)})
    u1 = unraveler({0: {0: no, 1: nv}})

    s11u = onebody_metric(t2)
    s11 = functoolz.compose(r1, s11u, u1)
    s = block_diag((s11, eye), (n1,))
    d = zero
    return s, d


def onebody_hessian_zeroth_order_diagonal(foo, fvv):
    eo = numpy.diagonal(foo)
    ev = numpy.diagonal(fvv)
    return broadcast_sum({0: -eo, 1: +ev})


def twobody_hessian_zeroth_order_diagonal(foo, fvv, t2):
    m1oo, m1vv = onebody_density(t2)
    ffoo = fancy_property(foo, m1oo)
    ffvv = fancy_property(fvv, m1vv)
    efo = numpy.diagonal(ffoo)
    efv = numpy.diagonal(ffvv)
    return broadcast_sum({0: -efo, 1: -efo, 2: -efv, 3: -efv})


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


def onebody_hessian(foo, fvv, goooo, goovv, govov, gvvvv, t2):
    m1oo, m1vv = onebody_density(t2)
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
            - einsum('jame,mkbc,ikec,jb...->ia...', govov, t2, t2, r1))

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
            + 1./4 * einsum('ijef,klef,klab,jb...->ia...', goovv, t2, t2, r1))

    return _a11, _b11


def mixed_upper_hessian(fov, gooov, govvv, t2):
    m1oo, m1vv = onebody_density(t2)
    fioo, fivv = _fancy_mixed_interaction(fov, gooov, govvv, m1oo, m1vv)

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

    def _b12(r2):
        return (
            + 1./2 * einsum('iamk,mlcd,klcd...->ia...', fioo, t2, r2)
            + 1./2 * einsum('iace,kled,klcd...->ia...', fivv, t2, r2)
            + einsum('lead,kice,klcd...->ia...', govvv, t2, r2)
            + einsum('ilmd,kmca,klcd...->ia...', gooov, t2, r2)
            - 1./4 * einsum('klma,micd,klcd...->ia...', gooov, t2, r2)
            - 1./4 * einsum('iecd,klea,klcd...->ia...', govvv, t2, r2))

    return _a12, _b12


def mixed_lower_hessian(fov, gooov, govvv, t2):
    m1oo, m1vv = onebody_density(t2)
    fioo, fivv = _fancy_mixed_interaction(fov, gooov, govvv, m1oo, m1vv)

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

    return _a21, _b21


def twobody_hessian(foo, fvv, goooo, govov, gvvvv, t2):
    m1oo, m1vv = onebody_density(t2)
    ffoo = fancy_property(foo, m1oo)
    ffvv = fancy_property(fvv, m1vv)
    fgoooo, fgovov, fgvvvv = _fancy_repulsion(
            ffoo, ffvv, goooo, govov, gvvvv, m1oo, m1vv)

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

    return _a22, _b22


def onebody_metric(t2):
    m1oo, m1vv = onebody_density(t2)

    def _s11(r1):
        return (
            + einsum('ij,ja...->ia...', m1oo, r1)
            - einsum('ab,ib...->ia...', m1vv, r1))

    return _s11


# Private
def _fancy_mixed_interaction(fov, gooov, govvv, m1oo, m1vv):
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


def _fancy_repulsion(ffoo, ffvv, goooo, govov, gvvvv, m1oo, m1vv):
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

import numpy
import scipy

from ..math import einsum
from ..math import transform
from ..math import broadcast_sum
from ..math.asym import antisymmetrizer_product as asm


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

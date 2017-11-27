import numpy
from ..math import einsum
from ..math.asym import antisymmetrizer_product as asm


def t_d1(pov, m1oo, m1vv):
    return (
        + einsum('...ie,ea->ia...', pov, m1vv)
        - einsum('im,...ma->ia...', m1oo, pov))


def t_d2(poo, pvv, t2):
    return (
        + asm('2/3')(
              einsum('...ac,ijcb->ijab...', pvv, t2))
        - asm('0/1')(
              einsum('...ik,kjab->ijab...', poo, t2)))


def a_d1d1_rf(hoo, hvv, goooo, goovv, govov, gvvvv, m1oo, m1vv, m2oooo,
              m2oovv, m2ovov, m2vvvv):
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


def b_d1d1_rf(goooo, goovv, govov, gvvvv, m2oooo, m2oovv, m2ovov, m2vvvv):

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


def s_d1d1_rf(m1oo, m1vv):

    def _sigma(r1):
        return (
            + einsum('ij,ja...->ia...', m1oo, r1)
            - einsum('ab,ib...->ia...', m1vv, r1))

    return _sigma


def a_d2d2_rf(foo, fvv, goooo, govov, gvvvv):

    def _sigma(r2):
        return (
            + asm('2/3')(einsum('ac,ijcb...->ijab...', fvv, r2))
            - asm('0/1')(einsum('ik,kjab...->ijab...', foo, r2))
            + 1./2 * einsum('abcd,ijcd...->ijab...', gvvvv, r2)
            + 1./2 * einsum('ijkl,klab...->ijab...', goooo, r2)
            - asm('0/1|2/3')(einsum('jcla,ilcb...->ijab...', govov, r2)))

    return _sigma


def mixed_interaction(fov, gooov, govvv):
    no, nv = fov.shape
    io = numpy.eye(no)
    iv = numpy.eye(nv)
    ioo = (+ einsum('ik,la->iakl', io, fov)
           - einsum('ilka->iakl', gooov))
    ivv = (- einsum('ac,id->iadc', iv, fov)
           + einsum('icad->iadc', govvv))
    return ioo, ivv


def a_d1d2_rf(fov, gooov, govvv, t2):
    ioo, ivv = mixed_interaction(fov, gooov, govvv)

    def _sigma(r2):
        return (
            + 1./2 * einsum('lacd,ilcd...->ia...', govvv, r2)
            + 1./2 * einsum('klid,klad...->ia...', gooov, r2)
            + 1./2 * einsum('iakm,mlcd,klcd...->ia...', ioo, t2, r2)
            - 1./2 * einsum('iaec,kled,klcd...->ia...', ivv, t2, r2)
            + einsum('mcae,mled,ilcd...->ia...', govvv, t2, r2)
            + einsum('imke,mled,klad...->ia...', gooov, t2, r2)
            + 1./4 * einsum('mnla,mncd,ilcd...->ia...', gooov, t2, r2)
            + 1./4 * einsum('idef,klef,klad...->ia...', govvv, t2, r2))

    return _sigma


def b_d1d2_rf(fov, gooov, govvv, t2):
    ioo, ivv = mixed_interaction(fov, gooov, govvv)

    def _sigma(r2):
        return (
            + 1./2 * einsum('iamk,mlcd,klcd...->ia...', ioo, t2, r2)
            - 1./2 * einsum('iace,kled,klcd...->ia...', ivv, t2, r2)
            + einsum('lead,kice,klcd...->ia...', govvv, t2, r2)
            + einsum('ilmd,kmca,klcd...->ia...', gooov, t2, r2)
            - 1./4 * einsum('klma,micd,klcd...->ia...', gooov, t2, r2)
            - 1./4 * einsum('iecd,klea,klcd...->ia...', govvv, t2, r2))

    return _sigma


def a_d1d2_lf(fov, gooov, govvv, t2):
    ioo, ivv = mixed_interaction(fov, gooov, govvv)

    def _sigma(r1):
        return (
            + asm('0/1')(
                einsum('jcab,ic...->ijab...', govvv, r1))
            + asm('2/3')(
                einsum('ijkb,ka...->ijab...', gooov, r1))
            + asm('0/1')(
                einsum('kcim,mjab,kc...->ijab...', ioo, t2, r1))
            - asm('2/3')(
                einsum('kcea,ijeb,kc...->ijab...', ivv, t2, r1))
            + asm('0/1|2/3')(
                einsum('mace,mjeb,ic...->ijab...', govvv, t2, r1))
            + asm('0/1|2/3')(
                einsum('kmie,mjeb,ka...->ijab...', gooov, t2, r1))
            + 1./2 * asm('0/1')(
                einsum('mnjc,mnab,ic...->ijab...', gooov, t2, r1))
            + 1./2 * asm('2/3')(
                einsum('kbef,ijef,ka...->ijab...', govvv, t2, r1)))

    return _sigma


def b_d1d2_lf(fov, gooov, govvv, t2):
    ioo, ivv = mixed_interaction(fov, gooov, govvv)

    def _sigma(r1):
        return (
            + asm('0/1')(
                einsum('kcmi,mjab,kc...->ijab...', ioo, t2, r1))
            - asm('2/3')(
                einsum('kcae,ijeb,kc...->ijab...', ivv, t2, r1))
            + asm('0/1|2/3')(
                einsum('jecb,ikae,kc...->ijab...', govvv, t2, r1))
            + asm('0/1|2/3')(
                einsum('kjmb,imac,kc...->ijab...', gooov, t2, r1))
            - einsum('ijmc,mkab,kc...->ijab...', gooov, t2, r1)
            - einsum('keab,ijec,kc...->ijab...', govvv, t2, r1))

    return _sigma
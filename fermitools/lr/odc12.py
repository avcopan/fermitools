import numpy
import scipy

from ..math import einsum
from ..math import transform
from ..math import broadcast_sum
from ..math.asym import antisymmetrizer_product as asm
from .ocepa0 import t_d1
from .ocepa0 import t_d2
from .ocepa0 import a_d1d1_rf
from .ocepa0 import b_d1d1_rf
from .ocepa0 import s_d1d1_rf
from .ocepa0 import a_d2d2_rf as cepa_a_d2d2_rf


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


def a_d1d2_rf(gooov, govvv, fioo, fivv, t2):

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


def b_d1d2_rf(gooov, govvv, fioo, fivv, t2):

    def _sigma(r2):
        return (
            + 1./2 * einsum('iamk,mlcd,klcd...->ia...', fioo, t2, r2)
            + 1./2 * einsum('iace,kled,klcd...->ia...', fivv, t2, r2)
            + einsum('lead,kice,klcd...->ia...', govvv, t2, r2)
            + einsum('ilmd,kmca,klcd...->ia...', gooov, t2, r2)
            - 1./4 * einsum('klma,micd,klcd...->ia...', gooov, t2, r2)
            - 1./4 * einsum('iecd,klea,klcd...->ia...', govvv, t2, r2))

    return _sigma


def a_d1d2_lf(gooov, govvv, fioo, fivv, t2):

    def _sigma(r1):
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

    return _sigma


def b_d1d2_lf(gooov, govvv, fioo, fivv, t2):

    def _sigma(r1):
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

    return _sigma


def a_d2d2_rf(ffoo, ffvv, goooo, govov, gvvvv, fgoooo, fgovov, fgvvvv, t2):
    cepa_a_d2d2_ = cepa_a_d2d2_rf(
            foo=+ffoo, fvv=-ffvv, goooo=goooo, govov=govov, gvvvv=gvvvv)

    def _sigma(r2):
        return (
            + cepa_a_d2d2_(r2)
            + 1./2 * asm('2/3')(
                einsum('afec,ijeb,klfd,klcd...->ijab...', fgvvvv, t2, t2, r2))
            + 1./2 * asm('2/3')(
                einsum('kame,ijeb,mlcd,klcd...->ijab...', fgovov, t2, t2, r2))
            + 1./2 * asm('0/1')(
                einsum('meic,mjab,kled,klcd...->ijab...', fgovov, t2, t2, r2))
            + 1./2 * asm('0/1')(
                einsum('mkin,mjab,nlcd,klcd...->ijab...', fgoooo, t2, t2, r2)))

    return _sigma


def b_d2d2_rf(fgoooo, fgovov, fgvvvv, t2):

    def _sigma(r2):
        return (
            + 1./2 * asm('2/3')(
                einsum('acef,ijeb,klfd,klcd...->ijab...', fgvvvv, t2, t2, r2))
            + 1./2 * asm('2/3')(
                einsum('nake,ijeb,nlcd,klcd...->ijab...', fgovov, t2, t2, r2))
            + 1./2 * asm('0/1')(
                einsum('mcif,mjab,klfd,klcd...->ijab...', fgovov, t2, t2, r2))
            + 1./2 * asm('0/1')(
                einsum('mnik,mjab,nlcd,klcd...->ijab...', fgoooo, t2, t2, r2)))

    return _sigma


__all__ = [
        'fancy_repulsion', 'fancy_mixed_interaction', 't_d1', 't_d2',
        'a_d1d1_rf', 'b_d1d1_rf', 's_d1d1_rf', 'a_d1d2_rf', 'b_d1d2_rf',
        'a_d1d2_lf', 'b_d1d2_lf', 'a_d2d2_rf', 'b_d2d2_rf']

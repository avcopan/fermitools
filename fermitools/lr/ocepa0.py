import numpy
from toolz import functoolz

from ..math import einsum
from ..math import broadcast_sum
from ..math import raveler, unraveler
from ..math.asym import antisymmetrizer_product as asm
from ..math.asym import megaraveler, megaunraveler
from ..math.sigma import eye
from ..math.sigma import zero
from ..math.sigma import bmat
from ..math.sigma import block_diag


# Public
def hessian_zeroth_order_diagonal(foo, fvv):
    r1 = raveler({0: (0, 1)})
    r2 = megaraveler({0: ((0, 1), (2, 3))})

    ad1u = onebody_hessian_zeroth_order_diagonal(foo, fvv)
    ad2u = twobody_hessian_zeroth_order_diagonal(foo, fvv)
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
    pg1 = r1(onebody_property_gradient(pov, t2))
    pg2 = r2(twobody_property_gradient(poo, pvv, t2))
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
    a22u, b22u = twobody_hessian(foo, fvv, goooo, govov, gvvvv)
    a11 = functoolz.compose(r1, a11u, u1)
    b11 = functoolz.compose(r1, b11u, u1)
    a12 = functoolz.compose(r1, a12u, u2)
    b12 = functoolz.compose(r1, b12u, u2)
    a21 = functoolz.compose(r2, a21u, u1)
    b21 = functoolz.compose(r2, b21u, u1)
    a22 = functoolz.compose(r2, a22u, u2)
    b22 = b22u
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


def twobody_hessian_zeroth_order_diagonal(foo, fvv):
    eo = numpy.diagonal(foo)
    ev = numpy.diagonal(fvv)
    return broadcast_sum({0: -eo, 1: -eo, 2: +ev, 3: +ev})


def onebody_property_gradient(pov, t2):
    return (
        - einsum('...ia->ia...', pov)
        + 1./2 * einsum('...ie,mnec,mnac->ia...', pov, t2, t2)
        + 1./2 * einsum('...ma,ikef,mkef->ia...', pov, t2, t2))


def twobody_property_gradient(poo, pvv, t2):
    return (
        + asm('2/3')(
              einsum('...ac,ijcb->ijab...', pvv, t2))
        - asm('0/1')(
              einsum('...ik,kjab->ijab...', poo, t2)))


def onebody_hessian(foo, fvv, goooo, goovv, govov, gvvvv, t2):
    fcoo = (foo
            + 1./2 * einsum('imef,jmef->ij', goovv, t2)
            - 1./2 * einsum('im,mkef,jkef->ij', foo, t2, t2)
            - 1./2 * einsum('imjo,mkef,okef->ij', goooo, t2, t2)
            + 1./2 * einsum('iejf,mnec,mnfc->ij', govov, t2, t2)
            + 1./4 * einsum('imno,jmcd,nocd->ij', goooo, t2, t2)
            - einsum('iemf,mkec,jkfc->ij', govov, t2, t2))
    fcvv = (+ 1./2 * einsum('mnae,mnbe', goovv, t2)
            + 1./2 * einsum('ae,mnec,mnbc->ab', fvv, t2, t2)
            + 1./4 * einsum('aefg,klbe,klfg', gvvvv, t2, t2)
            - einsum('nema,mkec,nkbc->ab', govov, t2, t2))
    fsoo = (fcoo + numpy.transpose(fcoo)) / 2.
    fsvv = (fcvv + numpy.transpose(fcvv)) / 2.

    def _a11(r1):
        return (
            + einsum('ab,ib...->ia...', fvv, r1)
            - einsum('ab,ib...->ia...', fsvv, r1)
            - einsum('ij,ja...->ia...', fsoo, r1)
            - einsum('jaib,jb...->ia...', govov, r1)
            - 1./2 * einsum('ab,ikef,jkef,jb...->ia...', fvv, t2, t2, r1)
            + 1./2 * einsum('ij,mnac,mnbc,jb...->ia...', foo, t2, t2, r1)
            + 1./2 * einsum('ibje,mnac,mnec,jb...->ia...', govov, t2, t2, r1)
            + 1./2 * einsum('jaie,mnbc,mnec,jb...->ia...', govov, t2, t2, r1)
            + 1./2 * einsum('aebf,mnec,mnfc,ib...->ia...', gvvvv, t2, t2, r1)
            - 1./2 * einsum('manb,mkef,nkef,ib...->ia...', govov, t2, t2, r1)
            + 1./2 * einsum('janb,ikef,nkef,jb...->ia...', govov, t2, t2, r1)
            + 1./2 * einsum('maib,mkef,jkef,jb...->ia...', govov, t2, t2, r1)
            + 1./2 * einsum('manb,micd,njcd,jb...->ia...', govov, t2, t2, r1)
            + 1./2 * einsum('iejf,klae,klbf,jb...->ia...', govov, t2, t2, r1)
            - einsum('minj,nkac,mkbc,jb...->ia...', goooo, t2, t2, r1)
            - einsum('ibme,mkac,jkec,jb...->ia...', govov, t2, t2, r1)
            - einsum('jame,mkbc,ikec,jb...->ia...', govov, t2, t2, r1)
            - einsum('aebf,jkec,ikfc,jb...->ia...', gvvvv, t2, t2, r1))

    def _b11(r1):
        return (
            + einsum('ijab,jb...->ia...', goovv, r1)
            + einsum('jema,imbe,jb...->ia...', govov, t2, r1)
            + einsum('iemb,jmae,jb...->ia...', govov, t2, r1)
            + 1./2 * einsum('ijmn,mnab,jb...->ia...', goooo, t2, r1)
            + 1./2 * einsum('efab,ijef,jb...->ia...', gvvvv, t2, r1)
            + 1./2 * einsum('ijbe,mnec,mnac,jb...->ia...', goovv, t2, t2, r1)
            + 1./2 * einsum('jiae,mnec,mnbc,jb...->ia...', goovv, t2, t2, r1)
            - 1./2 * einsum('inab,jkef,nkef,jb...->ia...', goovv, t2, t2, r1)
            - 1./2 * einsum('mjab,ikef,mkef,jb...->ia...', goovv, t2, t2, r1)
            + 1./4 * einsum('mnab,ijcd,mncd,jb...->ia...', goovv, t2, t2, r1)
            + 1./4 * einsum('ijef,klef,klab,jb...->ia...', goovv, t2, t2, r1)
            - einsum('imbe,mkec,jkac,jb...->ia...', goovv, t2, t2, r1)
            - einsum('jmae,mkec,ikbc,jb...->ia...', goovv, t2, t2, r1))

    return _a11, _b11


def mixed_upper_hessian(fov, gooov, govvv, t2):
    ioo, ivv = _mixed_interaction(fov, gooov, govvv)

    def _a12(r2):
        return (
            + 1./2 * einsum('lacd,ilcd...->ia...', govvv, r2)
            + 1./2 * einsum('klid,klad...->ia...', gooov, r2)
            + 1./2 * einsum('iakm,mlcd,klcd...->ia...', ioo, t2, r2)
            - 1./2 * einsum('iaec,kled,klcd...->ia...', ivv, t2, r2)
            + einsum('mcae,mled,ilcd...->ia...', govvv, t2, r2)
            + einsum('imke,mled,klad...->ia...', gooov, t2, r2)
            + 1./4 * einsum('mnla,mncd,ilcd...->ia...', gooov, t2, r2)
            + 1./4 * einsum('idef,klef,klad...->ia...', govvv, t2, r2))

    def _b12(r2):
        return (
            + 1./2 * einsum('iamk,mlcd,klcd...->ia...', ioo, t2, r2)
            - 1./2 * einsum('iace,kled,klcd...->ia...', ivv, t2, r2)
            + einsum('lead,kice,klcd...->ia...', govvv, t2, r2)
            + einsum('ilmd,kmca,klcd...->ia...', gooov, t2, r2)
            - 1./4 * einsum('klma,micd,klcd...->ia...', gooov, t2, r2)
            - 1./4 * einsum('iecd,klea,klcd...->ia...', govvv, t2, r2))

    return _a12, _b12


def mixed_lower_hessian(fov, gooov, govvv, t2):
    ioo, ivv = _mixed_interaction(fov, gooov, govvv)

    def _a21(r1):
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

    def _b21(r1):
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

    return _a21, _b21


def twobody_hessian(foo, fvv, goooo, govov, gvvvv):

    def _a22(r2):
        return (
            + asm('2/3')(einsum('ac,ijcb...->ijab...', fvv, r2))
            - asm('0/1')(einsum('ik,kjab...->ijab...', foo, r2))
            + 1./2 * einsum('abcd,ijcd...->ijab...', gvvvv, r2)
            + 1./2 * einsum('ijkl,klab...->ijab...', goooo, r2)
            - asm('0/1|2/3')(einsum('jcla,ilcb...->ijab...', govov, r2)))

    _b22 = zero

    return _a22, _b22


def onebody_metric(t2):

    def _s11(r1):
        return (
            + r1
            - 1./2 * einsum('ikef,jkef,ja...->ia...', t2, t2, r1)
            - 1./2 * einsum('mnac,mnbc,ib...->ia...', t2, t2, r1))

    return _s11


# Private
def _mixed_interaction(fov, gooov, govvv):
    no, nv = fov.shape
    io = numpy.eye(no)
    iv = numpy.eye(nv)
    ioo = (+ einsum('ik,la->iakl', io, fov)
           - einsum('ilka->iakl', gooov))
    ivv = (- einsum('ac,id->iadc', iv, fov)
           + einsum('icad->iadc', govvv))
    return ioo, ivv

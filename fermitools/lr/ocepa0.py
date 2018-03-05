import numpy

from ..math import cast
from ..math import einsum
from ..math.asym import antisymmetrizer_product as asm


# Public
def onebody_hessian_zeroth_order_diagonal(foo, fvv):
    eo = numpy.diagonal(foo)
    ev = numpy.diagonal(fvv)
    return - cast(eo, 0, 2) + cast(ev, 1, 2)


def twobody_hessian_zeroth_order_diagonal(foo, fvv):
    eo = numpy.diagonal(foo)
    ev = numpy.diagonal(fvv)
    return (- cast(eo, 0, 4) - cast(eo, 1, 4)
            + cast(ev, 2, 4) + cast(ev, 3, 4))


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
        a11 = (
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
        return a11

    def _b11(r1):
        b11 = (
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
        return b11

    return _a11, _b11


def mixed_hessian(fov, gooov, govvv, t2):
    ioo, ivv = _mixed_interaction(fov, gooov, govvv)

    def _a12(r2):
        a12 = (
            + 1./2 * einsum('lacd,ilcd...->ia...', govvv, r2)
            + 1./2 * einsum('klid,klad...->ia...', gooov, r2)
            + 1./2 * einsum('iakm,mlcd,klcd...->ia...', ioo, t2, r2)
            - 1./2 * einsum('iaec,kled,klcd...->ia...', ivv, t2, r2)
            + einsum('mcae,mled,ilcd...->ia...', govvv, t2, r2)
            + einsum('imke,mled,klad...->ia...', gooov, t2, r2)
            + 1./4 * einsum('mnla,mncd,ilcd...->ia...', gooov, t2, r2)
            + 1./4 * einsum('idef,klef,klad...->ia...', govvv, t2, r2))
        return a12

    def _b12(r2):
        b12 = (
            + 1./2 * einsum('iamk,mlcd,klcd...->ia...', ioo, t2, r2)
            - 1./2 * einsum('iace,kled,klcd...->ia...', ivv, t2, r2)
            + einsum('lead,kice,klcd...->ia...', govvv, t2, r2)
            + einsum('ilmd,kmca,klcd...->ia...', gooov, t2, r2)
            - 1./4 * einsum('klma,micd,klcd...->ia...', gooov, t2, r2)
            - 1./4 * einsum('iecd,klea,klcd...->ia...', govvv, t2, r2))
        return b12

    def _a21(r1):
        a21 = asm('0/1|2/3')(
            + 1./2 * einsum('jcab,ic...->ijab...', govvv, r1)
            + 1./2 * einsum('ijkb,ka...->ijab...', gooov, r1)
            + 1./2 * einsum('kcim,mjab,kc...->ijab...', ioo, t2, r1)
            - 1./2 * einsum('kcea,ijeb,kc...->ijab...', ivv, t2, r1)
            + einsum('mace,mjeb,ic...->ijab...', govvv, t2, r1)
            + einsum('kmie,mjeb,ka...->ijab...', gooov, t2, r1)
            + 1./4 * einsum('mnjc,mnab,ic...->ijab...', gooov, t2, r1)
            + 1./4 * einsum('kbef,ijef,ka...->ijab...', govvv, t2, r1))
        return a21

    def _b21(r1):
        b21 = asm('0/1|2/3')(
            + 1./2 * einsum('kcmi,mjab,kc...->ijab...', ioo, t2, r1)
            - 1./2 * einsum('kcae,ijeb,kc...->ijab...', ivv, t2, r1)
            + einsum('jecb,ikae,kc...->ijab...', govvv, t2, r1)
            + einsum('kjmb,imac,kc...->ijab...', gooov, t2, r1)
            - 1./4 * einsum('ijmc,mkab,kc...->ijab...', gooov, t2, r1)
            - 1./4 * einsum('keab,ijec,kc...->ijab...', govvv, t2, r1))
        return b21

    return _a12, _b12, _a21, _b21


def twobody_hessian(foo, fvv, goooo, govov, gvvvv):

    def _a22(r2):
        a22 = asm('0/1|2/3')(
            + 1./2 * einsum('ac,ijcb...->ijab...', fvv, r2)
            - 1./2 * einsum('ik,kjab...->ijab...', foo, r2)
            + 1./8 * einsum('abcd,ijcd...->ijab...', gvvvv, r2)
            + 1./8 * einsum('ijkl,klab...->ijab...', goooo, r2)
            - einsum('jcla,ilcb...->ijab...', govov, r2))
        return a22

    return _a22


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
    ioo = numpy.ascontiguousarray(
            + einsum('ik,la->iakl', io, fov)
            - einsum('ilka->iakl', gooov))
    ivv = numpy.ascontiguousarray(
            - einsum('ac,id->iadc', iv, fov)
            + einsum('icad->iadc', govvv))
    return ioo, ivv

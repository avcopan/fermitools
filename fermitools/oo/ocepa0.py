import numpy
from ..math import einsum
from ..math.asym import antisymmetrizer_product as asm

# Old
from .omp2 import onebody_correlation_density


# New
def fock_xy(hxy, goxoy):
    return hxy + numpy.trace(goxoy, axis1=0, axis2=2)


def twobody_amplitude_gradient(goooo, goovv, govov, gvvvv, foo, fvv, t2):
    return (goovv
            + asm("2/3")(einsum('ac,ijcb->ijab', fvv, t2))
            - asm("0/1")(einsum('ki,kjab->ijab', foo, t2))
            + 1. / 2 * einsum("abcd,ijcd->ijab", gvvvv, t2)
            + 1. / 2 * einsum("klij,klab->ijab", goooo, t2)
            - asm("0/1|2/3")(einsum("kaic,jkbc->ijab", govov, t2)))


def orbital_gradient(fov, gooov, govvv, t2):
    return (
            + einsum('ia->ia', fov)
            - 1./2 * einsum('ma,ikcd,mkcd->ia', fov, t2, t2)
            - 1./2 * einsum('ie,klac,klec->ia', fov, t2, t2)
            - 1./2 * einsum('mnie,mnae->ia', gooov, t2)
            + 1./2 * einsum('maef,mief->ia', govvv, t2)
            - 1./2 * einsum('mina,mkcd,nkcd->ia', gooov, t2, t2)
            + 1./4 * einsum('mkna,mkcd,nicd->ia', gooov, t2, t2)
            - 1./2 * einsum('ifea,klec,klfc->ia', govvv, t2, t2)
            + 1./4 * einsum('ifec,klec,klfa->ia', govvv, t2, t2)
            - einsum('mfae,ikfc,mkec->ia', govvv, t2, t2)
            + einsum('mine,nkac,mkec->ia', gooov, t2, t2))


# Old
__all__ = [
        'onebody_correlation_density',
        'twobody_amplitude_gradient', 'electronic_energy', 'orbital_gradient']


def fock_oo(hoo, goooo):
    return fock_xy(hoo, goooo)


def fock_vv(hvv, govov):
    return fock_xy(hvv, govov)


def fock_ov(hov, gooov):
    return fock_xy(hov, gooov)


def twobody_cumulant_oooo(t2):
    return 1./2 * einsum('ijcd,klcd->ijkl', t2, t2)


def twobody_cumulant_oovv(t2):
    return t2


def twobody_cumulant_ovov(t2):
    return -einsum('ikac,jkbc->jaib', t2, t2)


def twobody_cumulant_vvvv(t2):
    return 1./2 * einsum('klab,klcd->abcd', t2, t2)


def twobody_moment_oooo(dm1oo, cm1oo, k2oooo):
    return (k2oooo
            + asm("2/3")(
                + einsum('ik,jl->ijkl', cm1oo, dm1oo)
                + einsum('ik,jl->ijkl', dm1oo, dm1oo + cm1oo)))


def twobody_moment_ovov(dm1oo, cm1vv, k2ovov):
    return (k2ovov + einsum('ij,ab->iajb', dm1oo, cm1vv))


def electronic_energy(hoo, hvv, goooo, goovv, govov, gvvvv, m1oo, m1vv,
                      m2oooo, m2oovv, m2ovov, m2vvvv):
    return (+ numpy.vdot(hoo, m1oo)
            + numpy.vdot(hvv, m1vv)
            + 1./4 * numpy.vdot(goooo, m2oooo)
            + 1./2 * numpy.vdot(goovv, m2oovv)
            + numpy.vdot(govov, m2ovov)
            + 1./4 * numpy.vdot(gvvvv, m2vvvv))

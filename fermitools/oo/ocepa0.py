import numpy
from .hf import fock_oo, fock_ov, fock_vv
from .omp2 import onebody_correlation_density
from ..math import einsum
from ..math.asym import antisymmetrizer_product as asm


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


def twobody_moment_oovv(k2oovv):
    return k2oovv


def twobody_moment_ovov(dm1oo, cm1vv, k2ovov):
    return (k2ovov + einsum('ij,ab->iajb', dm1oo, cm1vv))


def twobody_moment_vvvv(k2vvvv):
    return k2vvvv


def electronic_energy(hoo, hvv, goooo, goovv, govov, gvvvv, m1oo, m1vv,
                      m2oooo, m2oovv, m2ovov, m2vvvv):
    return (+ numpy.vdot(hoo, m1oo)
            + numpy.vdot(hvv, m1vv)
            + 1./4 * numpy.vdot(goooo, m2oooo)
            + 1./2 * numpy.vdot(goovv, m2oovv)
            + numpy.vdot(govov, m2ovov)
            + 1./4 * numpy.vdot(gvvvv, m2vvvv))


def orbital_gradient(hov, gooov, govvv, m1oo, m1vv, m2oooo, m2oovv, m2ovov,
                     m2vvvv):
    return (+ einsum('ma,im->ia', hov, m1oo)
            - einsum('ie,ae->ia', hov, m1vv)
            + 1./2 * einsum('mlna,mlni->ia', gooov, m2oooo)
            - 1./2 * einsum('mnie,mnae->ia', gooov, m2oovv)
            + einsum('mfae,mfie->ia', govvv, m2ovov)
            - einsum('mine,mane->ia', gooov, m2ovov)
            + 1./2 * einsum('maef,mief->ia', govvv, m2oovv)
            - 1./2 * einsum('ifed,afed->ia', govvv, m2vvvv))


def twobody_amplitude_gradient(goooo, goovv, govov, gvvvv, foo, fvv, t2):
    return (goovv
            + asm("2/3")(einsum('ac,ijcb->ijab', fvv, t2))
            - asm("0/1")(einsum('ki,kjab->ijab', foo, t2))
            + 1. / 2 * einsum("abcd,ijcd->ijab", gvvvv, t2)
            + 1. / 2 * einsum("klij,klab->ijab", goooo, t2)
            - asm("0/1|2/3")(einsum("kaic,jkbc->ijab", govov, t2)))


__all__ = [
        'onebody_correlation_density', 'twobody_cumulant_oooo',
        'twobody_cumulant_oovv', 'twobody_cumulant_ovov',
        'twobody_cumulant_vvvv', 'twobody_moment_oooo', 'twobody_moment_oovv',
        'twobody_moment_ovov', 'twobody_moment_vvvv',
        'twobody_amplitude_gradient', 'electronic_energy', 'fock_oo',
        'fock_ov', 'fock_vv', 'orbital_gradient', 'twobody_amplitude_gradient']

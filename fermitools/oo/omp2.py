import numpy
from .hf import fock_oo, fock_ov, fock_vv
from ..math import einsum
from ..math.asym import antisymmetrizer_product as asm


def onebody_correlation_density(t2):
    """ the one-body correlation density matrix

    :param t2: two-body amplitudes
    :type t2: numpy.ndarray

    :returns: occupied and virtual blocks of correlation density
    :rtype: (numpy.ndarray, numpy.ndarray)
    """
    m1oo = - 1./2 * einsum('jkab,ikab->ij', t2, t2)
    m1vv = + 1./2 * einsum('ijac,ijbc->ab', t2, t2)
    return m1oo, m1vv


def twobody_cumulant_oovv(t2):
    return t2


def twobody_moment_oooo(dm1oo, cm1oo):
    return asm("2/3")(
        + einsum('ik,jl->ijkl', cm1oo, dm1oo)
        + einsum('ik,jl->ijkl', dm1oo, dm1oo + cm1oo))


def twobody_moment_oovv(k2oovv):
    return k2oovv


def twobody_moment_ovov(dm1oo, cm1vv):
    return einsum('ij,ab->iajb', dm1oo, cm1vv)


def twobody_moment_vvvv(k2vvvv):
    return k2vvvv


def electronic_energy(hoo, hvv, goooo, goovv, govov, m1oo, m1vv, m2oooo,
                      m2oovv, m2ovov):
    return (+ numpy.vdot(hoo, m1oo)
            + numpy.vdot(hvv, m1vv)
            + 1./4 * numpy.vdot(goooo, m2oooo)
            + 1./2 * numpy.vdot(goovv, m2oovv)
            + numpy.vdot(govov, m2ovov))


def orbital_gradient(hov, gooov, govvv, m1oo, m1vv, m2oooo, m2oovv, m2ovov):
    return (+ einsum('ma,im->ia', hov, m1oo)
            - einsum('ie,ae->ia', hov, m1vv)
            + 1./2 * einsum('mlna,mlni->ia', gooov, m2oooo)
            - 1./2 * einsum('mnie,mnae->ia', gooov, m2oovv)
            + einsum('mfae,mfie->ia', govvv, m2ovov)
            - einsum('mine,mane->ia', gooov, m2ovov)
            + 1./2 * einsum('maef,mief->ia', govvv, m2oovv))


def twobody_amplitude_gradient(goovv, foo, fvv, t2):
    return (goovv
            + asm("2/3")(einsum('ac,ijcb->ijab', fvv, t2))
            - asm("0/1")(einsum('ki,kjab->ijab', foo, t2)))


__all__ = [
        'onebody_correlation_density', 'twobody_cumulant_oovv',
        'twobody_moment_oooo', 'twobody_moment_oovv', 'twobody_moment_ovov',
        'twobody_moment_vvvv', 'electronic_energy', 'fock_oo', 'fock_ov',
        'fock_vv', 'orbital_gradient', 'twobody_amplitude_gradient']

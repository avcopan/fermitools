import numpy
import scipy.linalg

from ..math import broadcast_sum
from ..math import transform
from ..math import einsum
from ..math.asym import antisymmetrizer_product as asm
from .ocepa0 import (twobody_cumulant_oooo, twobody_cumulant_oovv,
                     twobody_cumulant_ovov, twobody_cumulant_vvvv)
from .ocepa0 import electronic_energy
from .ocepa0 import orbital_gradient, twobody_amplitude_gradient


def onebody_correlation_density(t2):
    """ the one-body correlation density matrix

    :param t2: two-body amplitudes
    :type t2: numpy.ndarray

    :returns: occupied and virtual blocks of correlation density
    :rtype: (numpy.ndarray, numpy.ndarray)
    """
    doo = -1./2 * einsum('ikcd,jkcd->ij', t2, t2)
    dvv = -1./2 * einsum('klac,klbc->ab', t2, t2)
    ioo = numpy.eye(*doo.shape)
    ivv = numpy.eye(*dvv.shape)
    m1oo = -1./2 * ioo + numpy.real(scipy.linalg.sqrtm(doo + 1./4 * ioo))
    m1vv = +1./2 * ivv - numpy.real(scipy.linalg.sqrtm(dvv + 1./4 * ivv))
    return m1oo, m1vv


def _twobody_moment_xxxx(m1xx, k2xxxx):
    return k2xxxx + asm("2/3")(einsum('xz,yw->xyzw', m1xx, m1xx))


def twobody_moment_oooo(m1oo, k2oooo):
    return _twobody_moment_xxxx(m1oo, k2oooo)


def twobody_moment_oovv(k2oovv):
    return k2oovv


def twobody_moment_ovov(m1oo, m1vv, k2ovov):
    return k2ovov + einsum('ij,ab->iajb', m1oo, m1vv)


def twobody_moment_vvvv(m1vv, k2vvvv):
    return _twobody_moment_xxxx(m1vv, k2vvvv)


def fock_oo(hoo, goooo, govov, m1oo, m1vv):
    return (hoo
            + numpy.tensordot(goooo, m1oo, axes=((0, 2), (0, 1)))
            + numpy.tensordot(govov, m1vv, axes=((1, 3), (0, 1))))


def fock_ov(hov, gooov, govvv, m1oo, m1vv):
    return (hov
            + numpy.tensordot(gooov, m1oo, axes=((0, 2), (0, 1)))
            + numpy.tensordot(govvv, m1vv, axes=((1, 3), (0, 1))))


def fock_vv(hvv, govov, gvvvv, m1oo, m1vv):
    return (hvv
            + numpy.tensordot(govov, m1oo, axes=((0, 2), (0, 1)))
            + numpy.tensordot(gvvvv, m1vv, axes=((1, 3), (0, 1))))


def fancy_property(pxx, m1xx):
    """ p_p^q (d m^p_q / d t) -> fp_p^q (d k^px_qx / dt)

    The one-body operator p can have multiple components.  Its spin-orbital
    indices are assumed to be the last two axes of the array.

    :param pxx: occupied or virtual block of a one-body operator
    :type pxx: numpy.ndarray
    :param m1xx: occupied or virtual block of the density
    :type m1xx: numpy.ndarray

    :returns: the derivative trace intermediate
    :rtype: numpy.ndarray
    """
    nx, ux = scipy.linalg.eigh(m1xx)
    ax1, ax2 = pxx.ndim - 2, pxx.ndim - 1
    n1xx = broadcast_sum({ax1: nx, ax2: nx}) - 1
    tfpxx = transform(pxx, {ax1: ux, ax2: ux}) / n1xx
    fpxx = transform(tfpxx, {ax1: ux.T, ax2: ux.T})
    return fpxx


__all__ = [
        'onebody_correlation_density', 'twobody_cumulant_oooo',
        'twobody_cumulant_oovv', 'twobody_cumulant_ovov',
        'twobody_cumulant_vvvv', 'twobody_moment_oooo', 'twobody_moment_oovv',
        'twobody_moment_ovov', 'twobody_moment_vvvv', 'electronic_energy',
        'fock_oo', 'fock_ov', 'fock_vv', 'fancy_property', 'orbital_gradient',
        'twobody_amplitude_gradient']

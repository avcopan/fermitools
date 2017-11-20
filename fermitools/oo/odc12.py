import numpy
import scipy.linalg

from ..math import broadcast_sum
from ..math import transform
from ..math import einsum
from .ocepa0 import twobody_amplitude_gradient


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


__all__ = ['fancy_property', 'onebody_correlation_density',
           'twobody_amplitude_gradient']

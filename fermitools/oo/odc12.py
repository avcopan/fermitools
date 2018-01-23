import numpy
import scipy.linalg
import warnings

from ..math import expm
from ..math import einsum
from ..math import transform
from ..math import broadcast_sum
from ..math.spinorb import decompose_onebody
from ..math.spinorb import transform_onebody, transform_twobody

from .ocepa0 import twobody_amplitude_gradient


def solve(na, nb, h_ao, r_ao, c_guess, t2_guess, niter=50, r_thresh=1e-8,
          print_conv=True):
    no, _, nv, _ = t2_guess.shape

    ac, bc = c_guess
    t2 = t2_guess
    zoo = numpy.zeros((no, no))
    zvv = numpy.zeros((nv, nv))
    m1oo, m1vv = onebody_density(t2)

    for iteration in range(niter):
        aco, acv = numpy.split(ac, (na,), axis=1)
        bco, bcv = numpy.split(bc, (nb,), axis=1)
        co = (aco, bco)
        cv = (acv, bcv)
        hoo = transform_onebody(h_ao, (co, co))
        hov = transform_onebody(h_ao, (co, cv))
        hvv = transform_onebody(h_ao, (cv, cv))
        goooo = transform_twobody(r_ao, (co, co, co, co))
        gooov = transform_twobody(r_ao, (co, co, co, cv))
        goovv = transform_twobody(r_ao, (co, co, cv, cv))
        govov = transform_twobody(r_ao, (co, cv, co, cv))
        govvv = transform_twobody(r_ao, (co, cv, cv, cv))
        gvvvv = transform_twobody(r_ao, (cv, cv, cv, cv))
        # Orbital step
        m1oo, m1vv = onebody_density(t2)
        foo = fock_xy(hoo, goooo, govov, m1oo, m1vv)
        fov = fock_xy(hov, gooov, govvv, m1oo, m1vv)
        fvv = fock_xy(hvv, govov, gvvvv, m1oo, m1vv)
        eo = numpy.diagonal(foo)
        ev = numpy.diagonal(fvv)
        e1 = broadcast_sum({0: +eo, 1: -ev})
        r1 = orbital_gradient(fov, gooov, govvv, m1oo, m1vv, t2)
        t1 = r1 / e1
        a = numpy.bmat([[zoo, -t1], [+t1.T, zvv]])
        u = expm(a)
        au, bu = decompose_onebody(u, na=na, nb=nb)
        ac = numpy.dot(ac, au)
        bc = numpy.dot(bc, bu)
        c = (ac, bc)
        # Amplitude step
        ffoo = fancy_property(foo, m1oo)
        ffvv = fancy_property(fvv, m1vv)
        feo = numpy.diagonal(ffoo)
        fev = numpy.diagonal(ffvv)
        fe2 = broadcast_sum({0: +feo, 1: +feo, 2: +fev, 3: +fev})
        r2 = twobody_amplitude_gradient(
                goooo, goovv, govov, gvvvv, +ffoo, -ffvv, t2)
        t2 = t2 + r2 / fe2

        r1_max = numpy.amax(numpy.abs(r1))
        r2_max = numpy.amax(numpy.abs(r2))

        info = {'niter': iteration + 1, 'r1_max': r1_max, 'r2_max': r2_max}

        converged = r1_max < r_thresh and r2_max < r_thresh

        if print_conv:
            print(info, flush=True)

        if converged:
            break

    en_elec = electronic_energy(
            hoo, hvv, goooo, goovv, govov, gvvvv, m1oo, m1vv, t2)

    if not converged:
        warnings.warn("Did not converge!")

    return en_elec, c, t2, info


def fock_xy(hxy, goxoy, gxvyv, m1oo, m1vv):
    return (hxy
            + numpy.tensordot(goxoy, m1oo, axes=((0, 2), (0, 1)))
            + numpy.tensordot(gxvyv, m1vv, axes=((1, 3), (0, 1))))


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
    tfpxx = transform(pxx, (ux, ux)) / n1xx
    fpxx = transform(tfpxx, (ux.T, ux.T))
    return fpxx


def onebody_density(t2):
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
    m1oo = 1./2 * ioo + numpy.real(scipy.linalg.sqrtm(doo + 1./4 * ioo))
    m1vv = 1./2 * ivv - numpy.real(scipy.linalg.sqrtm(dvv + 1./4 * ivv))
    return m1oo, m1vv


def orbital_gradient(fov, gooov, govvv, m1oo, m1vv, t2):
    return (+ einsum('ma,im->ia', fov, m1oo)
            - einsum('ie,ae->ia', fov, m1vv)
            - 1./2 * einsum('mnie,mnae->ia', gooov, t2)
            + 1./2 * einsum('maef,mief->ia', govvv, t2)
            + 1./4 * einsum('mlna,mlcd,nicd->ia', gooov, t2, t2)
            - 1./4 * einsum('ifed,klaf,kled->ia', govvv, t2, t2)
            - einsum('mfae,ikfc,mkec->ia', govvv, t2, t2)
            + einsum('mine,nkac,mkec->ia', gooov, t2, t2))


def electronic_energy(hoo, hvv, goooo, goovv, govov, gvvvv, m1oo, m1vv, t2):
    foo = fock_xy(hoo, goooo, govov, m1oo, m1vv)
    fvv = fock_xy(hvv, govov, gvvvv, m1oo, m1vv)
    return (+ 1./2 * numpy.vdot(hoo + foo, m1oo)
            + 1./2 * numpy.vdot(hvv + fvv, m1vv)
            + 1./2 * numpy.vdot(goovv, t2)
            - einsum('iajb,jkac,ikbc', govov, t2, t2)
            + 1./8 * einsum('abcd,klab,klcd', gvvvv, t2, t2)
            + 1./8 * einsum('ijkl,ijcd,klcd', goooo, t2, t2))

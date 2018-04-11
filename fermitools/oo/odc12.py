import sys
import warnings
import numpy
import scipy.linalg

from .util import orbital_rotation
from .util import diis_extrapolator
from ..math import einsum
from ..math import cast
from ..math import transform
from ..math.spinorb import transform_onebody, transform_twobody

from .ocepa0 import twobody_amplitude_gradient


def compute_property(p_ao, co, cv, t2):
    poo = transform_onebody(p_ao, (co, co))
    pvv = transform_onebody(p_ao, (cv, cv))
    m1oo, m1vv = onebody_density(t2)
    mu = (numpy.tensordot(poo, m1oo, axes=((-2, -1), (0, 1))) +
          numpy.tensordot(pvv, m1vv, axes=((-2, -1), (0, 1))))
    print("First-order properties:")
    print(mu.round(12))
    return mu


def solve(h_ao, r_ao, co_guess, cv_guess, t2_guess, maxiter=50, rthresh=1e-8,
          diis_start=3, diis_nvec=20, print_conv=True):
    no, _, nv, _ = t2_guess.shape
    t1 = numpy.zeros((no, nv))
    t2 = t2_guess
    m1oo, m1vv = onebody_density(t2)

    trs = ()
    extrapolate = diis_extrapolator(start=diis_start, nvec=diis_nvec)

    for iteration in range(maxiter):
        co, cv = orbital_rotation(co_guess, cv_guess, t1)
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
        e1 = cast(eo, 0, 2) - cast(ev, 1, 2)
        r1 = orbital_gradient(fov, gooov, govvv, m1oo, m1vv, t2)
        t1 = t1 + r1 / e1
        # Amplitude step
        ffoo = fancy_property(foo, m1oo)
        ffvv = fancy_property(fvv, m1vv)
        feo = numpy.diagonal(ffoo)
        fev = numpy.diagonal(ffvv)
        fe2 = (+ cast(feo, 0, 4) + cast(feo, 1, 4)
               + cast(fev, 2, 4) + cast(fev, 3, 4))
        r2 = twobody_amplitude_gradient(
                goooo, goovv, govov, gvvvv, +ffoo, -ffvv, t2)
        t2 = t2 + r2 / fe2

        r1max = numpy.amax(numpy.abs(r1))
        r2max = numpy.amax(numpy.abs(r2))

        info = {'niter': iteration + 1, 'r1max': r1max, 'r2max': r2max}

        converged = r1max < rthresh and r2max < rthresh

        if print_conv:
            print(info)
            sys.stdout.flush()

        if converged:
            break

        (t1, t2), trs = extrapolate(t=(t1, t2), r=(r1, r2), trs=trs)

    en_elec = electronic_energy(
            hoo, hvv, goooo, goovv, govov, gvvvv, m1oo, m1vv, t2)

    if not converged:
        warnings.warn("Did not converge!")

    return en_elec, co, cv, t2, info


# The ODC-12 equations
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
    mx, ux = scipy.linalg.eigh(m1xx)
    ndim = pxx.ndim
    n1xx = cast(mx, ndim-2, ndim) + cast(mx, ndim-1, ndim) - 1
    tfpxx = transform(pxx, (ux, ux)) / n1xx
    uxt = numpy.ascontiguousarray(numpy.transpose(ux))
    fpxx = transform(tfpxx, (uxt, uxt))
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
    cfov = orbital_gradient_intermediate_xv(
            fxv=fov, gooxv=gooov, goxov=gooov, gxvvv=govvv, t2=t2, m1vv=m1vv)
    cfvo = orbital_gradient_intermediate_xo(
            fox=fov, gooox=gooov, goxvv=govvv, govxv=govvv, t2=t2, m1oo=m1oo)
    return numpy.transpose(cfvo) - cfov


def orbital_gradient_intermediate_xo(fox, gooox, goxvv, govxv, t2, m1oo):
    cfxo = (numpy.tensordot(fox, m1oo, axes=(0, 0))
            - 1./2 * einsum('mpef,imef->pi', goxvv, t2)
            - 1./4 * einsum('nomp,imcd,nocd->pi', gooox, t2, t2)
            - einsum('mfpe,mkec,ikfc->pi', govxv, t2, t2))
    return cfxo


def orbital_gradient_intermediate_xv(fxv, gooxv, goxov, gxvvv, t2, m1vv):
    cfxv = (numpy.dot(fxv, m1vv)
            + 1./2 * einsum('mnpe,mnae->pa', gooxv, t2)
            + 1./4 * einsum('pefg,klae,klfg->pa', gxvvv, t2, t2)
            - einsum('mpne,mkec,nkac->pa', goxov, t2, t2))
    return cfxv


def electronic_energy(hoo, hvv, goooo, goovv, govov, gvvvv, m1oo, m1vv, t2):
    foo = fock_xy(hoo, goooo, govov, m1oo, m1vv)
    fvv = fock_xy(hvv, govov, gvvvv, m1oo, m1vv)
    return (+ 1./2 * numpy.vdot(hoo + foo, m1oo)
            + 1./2 * numpy.vdot(hvv + fvv, m1vv)
            + 1./2 * numpy.vdot(goovv, t2)
            - einsum('iajb,jkac,ikbc', govov, t2, t2)
            + 1./8 * einsum('abcd,klab,klcd', gvvvv, t2, t2)
            + 1./8 * einsum('ijkl,ijcd,klcd', goooo, t2, t2))

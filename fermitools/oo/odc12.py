import numpy
import scipy.linalg
import warnings

from ..math import broadcast_sum
from ..math import transform
from ..math import einsum
from ..math.asym import antisymmetrizer_product as asm


def solve(h_aso, g_aso, c_guess, t2_guess, niter=50, r_thresh=1e-8):
    no, _, nv, _ = t2_guess.shape

    c = c_guess
    t2 = t2_guess
    zoo = numpy.zeros((no, no))
    zvv = numpy.zeros((nv, nv))
    m1oo, m1vv = onebody_density(t2)

    for iteration in range(niter):
        co, cv = numpy.split(c, (no,), axis=1)
        hoo = transform(h_aso, {0: co, 1: co})
        hov = transform(h_aso, {0: co, 1: cv})
        hvv = transform(h_aso, {0: cv, 1: cv})
        goooo = transform(g_aso, {0: co, 1: co, 2: co, 3: co})
        gooov = transform(g_aso, {0: co, 1: co, 2: co, 3: cv})
        goovv = transform(g_aso, {0: co, 1: co, 2: cv, 3: cv})
        govov = transform(g_aso, {0: co, 1: cv, 2: co, 3: cv})
        govvv = transform(g_aso, {0: co, 1: cv, 2: cv, 3: cv})
        gvvvv = transform(g_aso, {0: cv, 1: cv, 2: cv, 3: cv})
        # Orbital step
        m1oo, m1vv = onebody_density(t2)
        foo = fock_xy(hxy=hoo, goxoy=goooo, gxvyv=govov, m1oo=m1oo, m1vv=m1vv)
        fvv = fock_xy(hxy=hvv, goxoy=govov, gxvyv=gvvvv, m1oo=m1oo, m1vv=m1vv)
        eo = numpy.diagonal(foo)
        ev = numpy.diagonal(fvv)
        e1 = broadcast_sum({0: +eo, 1: -ev})
        r1 = orbital_gradient(hov, gooov, govvv, m1oo, m1vv, t2)
        t1 = r1 / e1
        a = numpy.bmat([[zoo, -t1], [+t1.T, zvv]])
        u = scipy.linalg.expm(a)
        c = numpy.dot(c, u)
        # Amplitude step
        ffoo = fancy_property(foo, m1oo)
        ffvv = fancy_property(fvv, m1vv)
        feo = numpy.diagonal(ffoo)
        fev = numpy.diagonal(ffvv)
        fe2 = broadcast_sum({0: +feo, 1: +feo, 2: +fev, 3: +fev})
        r2 = twobody_amplitude_gradient(
                goooo, goovv, govov, gvvvv, +ffoo, -ffvv, t2)
        t2 = t2 + r2 / fe2

        en_elec = electronic_energy(
                hoo, hvv, goooo, goovv, govov, gvvvv, m1oo, m1vv, t2)

        r1_max = numpy.amax(numpy.abs(r1))
        r2_max = numpy.amax(numpy.abs(r2))

        converged = r1_max < r_thresh and r2_max < r_thresh

        if converged:
            break

    info = {'niter': iteration + 1, 'r1_max': r1_max, 'r2_max': r2_max}

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
    tfpxx = transform(pxx, {ax1: ux, ax2: ux}) / n1xx
    fpxx = transform(tfpxx, {ax1: ux.T, ax2: ux.T})
    return fpxx


def twobody_amplitude_gradient(goooo, goovv, govov, gvvvv, foo, fvv, t2):
    return (goovv
            + asm("2/3")(einsum('ac,ijcb->ijab', fvv, t2))
            - asm("0/1")(einsum('ki,kjab->ijab', foo, t2))
            + 1. / 2 * einsum("abcd,ijcd->ijab", gvvvv, t2)
            + 1. / 2 * einsum("klij,klab->ijab", goooo, t2)
            - asm("0/1|2/3")(einsum("kaic,jkbc->ijab", govov, t2)))


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


def orbital_gradient(hov, gooov, govvv, m1oo, m1vv, t2):
    return (+ einsum('ma,im->ia', hov, m1oo)
            - einsum('ie,ae->ia', hov, m1vv)
            + einsum('mlna,mn,li->ia', gooov, m1oo, m1oo)
            - einsum('ifed,ae,fd->ia', govvv, m1vv, m1vv)
            + einsum('mfae,mi,fe->ia', govvv, m1oo, m1vv)
            - einsum('mine,mn,ae->ia', gooov, m1oo, m1vv)
            - 1./2 * einsum('mnie,mnae->ia', gooov, t2)
            + 1./2 * einsum('maef,mief->ia', govvv, t2)
            + 1./4 * einsum('mlna,mlcd,nicd->ia', gooov, t2, t2)
            - 1./4 * einsum('ifed,klaf,kled->ia', govvv, t2, t2)
            - einsum('mfae,ikfc,mkec->ia', govvv, t2, t2)
            + einsum('mine,nkac,mkec->ia', gooov, t2, t2))


def electronic_energy(hoo, hvv, goooo, goovv, govov, gvvvv, m1oo, m1vv, t2):
    return (+ numpy.vdot(hoo, m1oo)
            + numpy.vdot(hvv, m1vv)
            + 1./2 * einsum('ikjl,ij,kl', goooo, m1oo, m1oo)
            + 1./2 * einsum('acbd,ab,cd', gvvvv, m1vv, m1vv)
            + einsum('iajb,ij,ab', govov, m1oo, m1vv)
            + 1./2 * numpy.vdot(goovv, t2)
            - einsum('iajb,jkac,ikbc', govov, t2, t2)
            + 1./8 * einsum('abcd,klab,klcd', gvvvv, t2, t2)
            + 1./8 * einsum('ijkl,ijcd,klcd', goooo, t2, t2))

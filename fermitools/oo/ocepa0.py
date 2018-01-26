import numpy
import warnings
import sys

from .util import orbital_rotation
from ..math import einsum
from ..math import broadcast_sum
from ..math.asym import antisymmetrizer_product as asm
from ..math.spinorb import transform_onebody, transform_twobody


def solve(h_ao, r_ao, co_guess, cv_guess, t2_guess, niter=50, rthresh=1e-8,
          print_conv=True):
    no, _, nv, _ = t2_guess.shape
    t1 = numpy.zeros((no, nv))
    t2 = t2_guess

    for iteration in range(niter):
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
        foo = fock_xy(hoo, goooo)
        fov = fock_xy(hov, gooov)
        fvv = fock_xy(hvv, govov)
        eo = numpy.diagonal(foo)
        ev = numpy.diagonal(fvv)
        e1 = broadcast_sum({0: +eo, 1: -ev})
        r1 = orbital_gradient(fov, gooov, govvv, t2)
        t1 = t1 + r1 / e1
        # Amplitude step
        e2 = broadcast_sum({0: +eo, 1: +eo, 2: -ev, 3: -ev})
        r2 = twobody_amplitude_gradient(
                goooo, goovv, govov, gvvvv, foo, fvv, t2)
        t2 += r2 / e2

        r1max = numpy.amax(numpy.abs(r1))
        r2max = numpy.amax(numpy.abs(r2))

        info = {'niter': iteration + 1, 'r1max': r1max, 'r2max': r2max}

        converged = r1max < rthresh and r2max < rthresh

        if print_conv:
            print(info)
            sys.stdout.flush()

        if converged:
            break

    en_elec = electronic_energy(hoo, hvv, goooo, goovv, govov, gvvvv, t2)

    if not converged:
        warnings.warn("Did not converge!")

    return en_elec, co, cv, t2, info


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
    return (+ fov
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


def electronic_energy(hoo, hvv, goooo, goovv, govov, gvvvv, t2):
    foo = fock_xy(hoo, goooo)
    fvv = fock_xy(hvv, govov)
    return (+ 1./2 * numpy.trace(hoo + foo)
            - 1./2 * einsum('ij,jkab,ikab', foo, t2, t2)
            + 1./2 * einsum('ab,ijac,ijbc', fvv, t2, t2)
            + 1./2 * numpy.vdot(goovv, t2)
            - einsum('iajb,jkac,ikbc', govov, t2, t2)
            + 1./8 * einsum('abcd,klab,klcd', gvvvv, t2, t2)
            + 1./8 * einsum('ijkl,ijcd,klcd', goooo, t2, t2))


def onebody_density(t2):
    """ the one-body density matrix

    :param t2: two-body amplitudes
    :type t2: numpy.ndarray

    :returns: occupied and virtual blocks of correlation density
    :rtype: (numpy.ndarray, numpy.ndarray)
    """
    no, _, nv, _ = t2.shape
    m1oo = - 1./2 * einsum('jkab,ikab->ij', t2, t2) + numpy.eye(no)
    m1vv = + 1./2 * einsum('ijac,ijbc->ab', t2, t2)
    return m1oo, m1vv

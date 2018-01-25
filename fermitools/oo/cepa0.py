import numpy
import warnings
import sys

from ..math import einsum
from ..math import broadcast_sum
from ..math.asym import antisymmetrizer_product as asm


def solve_diis(foo, fvv, goooo, goovv, govov, gvvvv, t2_guess, niter=50,
               r_thresh=1e-8, print_conv=True):
    from ..math.diis import extrapolate

    eo = numpy.diagonal(foo)
    ev = numpy.diagonal(fvv)
    e2 = broadcast_sum({0: +eo, 1: +eo, 2: -ev, 3: -ev})
    t2 = t2_guess

    t2s = []
    r2s = []

    for iteration in range(niter):
        r2 = twobody_amplitude_gradient(
                goooo, goovv, govov, gvvvv, foo, fvv, t2)
        t2 += r2 / e2

        r2_max = numpy.amax(numpy.abs(r2))

        info = {'niter': iteration + 1, 'r2_max': r2_max}

        converged = r2_max < r_thresh

        if print_conv:
            print(info)
            sys.stdout.flush()

        if converged:
            break

        r2s.append(r2)
        t2s.append(t2)
        t2 = extrapolate(t2s, r2s) if len(t2s) > 3 else t2

    en_corr = numpy.vdot(goovv, t2) / 4.

    if not converged:
        warnings.warn("Did not converge!")

    return en_corr, t2, info


def solve(foo, fvv, goooo, goovv, govov, gvvvv, t2_guess, niter=50,
          r_thresh=1e-8, print_conv=True):

    eo = numpy.diagonal(foo)
    ev = numpy.diagonal(fvv)
    e2 = broadcast_sum({0: +eo, 1: +eo, 2: -ev, 3: -ev})
    t2 = t2_guess

    for iteration in range(niter):
        r2 = twobody_amplitude_gradient(
                goooo, goovv, govov, gvvvv, foo, fvv, t2)
        t2 += r2 / e2

        r2_max = numpy.amax(numpy.abs(r2))

        info = {'niter': iteration + 1, 'r2_max': r2_max}

        converged = r2_max < r_thresh

        if print_conv:
            print(info)
            sys.stdout.flush()

        if converged:
            break

    en_corr = numpy.vdot(goovv, t2) / 4.

    if not converged:
        warnings.warn("Did not converge!")

    return en_corr, t2, info


def fock_xy(hxy, goxoy):
    return hxy + numpy.trace(goxoy, axis1=0, axis2=2)


def twobody_amplitude_gradient(goooo, goovv, govov, gvvvv, foo, fvv, t2):
    return (goovv
            + asm("2/3")(einsum('ac,ijcb->ijab', fvv, t2))
            - asm("0/1")(einsum('ki,kjab->ijab', foo, t2))
            + 1. / 2 * einsum("abcd,ijcd->ijab", gvvvv, t2)
            + 1. / 2 * einsum("klij,klab->ijab", goooo, t2)
            - asm("0/1|2/3")(einsum("kaic,jkbc->ijab", govov, t2)))

import numpy
from ..math import einsum


def fock_block(hxy, goxoy, m1oo, gxvyv=None, m1vv=None):
    fxy = hxy + numpy.tensordot(goxoy, m1oo, axes=((0, 2), (0, 1)))
    return (fxy if m1vv is None else
            fxy + numpy.tensordot(gxvyv, m1vv, axes=((1, 3), (0, 1))))


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


def electronic_energy(hoo, hvv, goooo, goovv, govov, gvvvv, m1oo, m1vv,
                      m2oooo, m2oovv, m2ovov, m2vvvv):
    return (+ numpy.vdot(hoo, m1oo)
            + numpy.vdot(hvv, m1vv)
            + 1./4 * numpy.vdot(goooo, m2oooo)
            + 1./2 * numpy.vdot(goovv, m2oovv)
            + numpy.vdot(govov, m2ovov)
            + 1./4 * numpy.vdot(gvvvv, m2vvvv))

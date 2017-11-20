import numpy


def fock_block(hxy, goxoy, m1oo, gxvyv=None, m1vv=None):
    fxy = hxy + numpy.tensordot(goxoy, m1oo, axes=((0, 2), (0, 1)))
    return (fxy if m1vv is None else
            fxy + numpy.tensordot(gxvyv, m1vv, axes=((1, 3), (0, 1))))

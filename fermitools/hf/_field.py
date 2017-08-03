import numpy


def coulomb(g, ad, bd):
    """coulomb repulsion matrix

    :param g: electron repulsion integrals, in bra-ket notation
    :type g: numpy.ndarray
    :param ad: hartree-fock alpha density matrix
    :type ad: numpy.ndarray
    :param bd: hatree-fock beta density matrix
    :type bd: numpy.ndarray

    :rtype: numpy.ndarray
    """
    return numpy.tensordot(g, ad + bd, axes=((1, 3), (1, 0)))


def exchange(g, d):
    """alpha or beta exchange matrix

    :param g: electron repulsion integrals, in bra-ket notation
    :type g: numpy.ndarray
    :param d: alpha or beta density matrix
    :type d: numpy.ndarray

    :rtype: numpy.ndarray
    """
    return numpy.tensordot(g, d, axes=((1, 2), (1, 0)))

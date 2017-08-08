"""Hartree-Fock spin-integrated coulomb and exchange fields."""
import numpy


def coulomb(g, d):
    """coulomb repulsion matrix

    :param g: electron repulsion integrals, in bra-ket notation
    :type g: numpy.ndarray
    :param d: density matrix
    :type d: numpy.ndarray

    :rtype: numpy.ndarray
    """
    return numpy.tensordot(g, d, axes=((1, 3), (1, 0)))


def exchange(g, d):
    """alpha or beta exchange matrix

    :param g: electron repulsion integrals, in bra-ket notation
    :type g: numpy.ndarray
    :param d: alpha or beta density matrix
    :type d: numpy.ndarray

    :rtype: numpy.ndarray
    """
    return numpy.tensordot(g, d, axes=((1, 2), (1, 0)))

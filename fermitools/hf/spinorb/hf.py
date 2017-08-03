"""spin-orbital Hartree-Fock"""
import numpy


def fock(h, g, d):
    """fock matrix

    :param h: spin-AO core hamiltonian
    :type h: numpy.ndarray
    :param g: antisymmetrized (!) spin-AO electron repulsion integrals
    :type g: numpy.ndarray
    :param d: hartree-fock spin-orbital density matrix
    :type d: numpy.ndarray

    :rtype: numpy.ndarray
    """
    w = numpy.tensordot(g, d, axes=((1, 3), (1, 0)))
    return h + w


def energy(h, f, d):
    """electronic energy, not including nuclear repulsion

    :param h: spin-AO core hamiltonian
    :type h: numpy.ndarray
    :param f: spin-AO fock matrix
    :type f: numpy.ndarray
    :param d: hartree-fock spin-orbital density matrix
    :type d: numpy.ndarray

    :rtype: float
    """
    return 1. / 2 * numpy.tensordot(h + f, d, axes=((0, 1), (1, 0)))

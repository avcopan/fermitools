"""restricted closed-shell Hartree-Fock"""
import numpy
from ._field import coulomb, exchange


def fock(h, g, d):
    """fock matrix

    :param h: core hamiltonian
    :type h: numpy.ndarray
    :param g: electron repulsion integrals, in bra-ket notation
    :type g: numpy.ndarray
    :param d: hartree-fock density matrix
    :type d: numpy.ndarray

    :rtype: numpy.ndarray
    """
    j = coulomb(g, d=2.*d)
    k = exchange(g, d=d)
    return h + j - k


def energy(h, f, d):
    """electronic energy, not including nuclear repulsion

    :param h:core hamiltonian
    :type h: numpy.ndarray
    :param f: fock matrix
    :type f: numpy.ndarray
    :param d: hartree-fock density matrix
    :type d: numpy.ndarray

    :rtype: float
    """
    return numpy.vdot(h + f, d)

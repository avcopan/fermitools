"""unrestricted open-shell Hartree-Fock"""
import numpy
from ._field import coulomb, exchange


def fock(h, g, ad, bd):
    """alpha and beta fock matrices

    :param h: core hamiltonian
    :type h: numpy.ndarray
    :param g: electron repulsion integrals, in bra-ket notation
    :type g: numpy.ndarray
    :param ad: hartree-fock alpha density matrix
    :type ad: numpy.ndarray
    :param bd: hatree-fock beta density matrix
    :type bd: numpy.ndarray

    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    j = coulomb(g=g, d=ad+bd)
    ak = exchange(g=g, d=ad)
    bk = exchange(g=g, d=bd)
    return h + j - ak, h + j - bk


def energy(h, af, bf, ad, bd):
    """electronic energy, not including nuclear repulsion

    :param h: core hamiltonian
    :type h: numpy.ndarray
    :param af: alpha fock matrix
    :type af: numpy.ndarray
    :param bf: beta fock matrix
    :type bf: numpy.ndarray
    :param ad: hartree-fock alpha density matrix
    :type ad: numpy.ndarray
    :param bd: hatree-fock beta density matrix
    :type bd: numpy.ndarray

    :rtype: float
    """
    return (numpy.vdot(h + af, ad) + numpy.vdot(h + bf, bd)) / 2.

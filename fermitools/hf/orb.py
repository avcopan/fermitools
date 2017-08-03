import numpy
import scipy.linalg as spla


def density(n, c):
    """hf density matrix

    :param n: number of occupied orbitals
    :type n: int
    :param c: orbital coefficients
    :type c: numpy.ndarray

    :rtype: numpy.ndarray
    """
    c_o = c[:, :n]
    return numpy.dot(c_o, c_o.T)


def coefficients(s, f):
    """orbital coefficients

    :param s: basis function overlap matrix
    :type s: numpy.ndarray
    :param f: fock matrix
    :type f: numpy.ndarray

    :rtype: numpy.ndarray
    """
    _, c = spla.eigh(f, b=s)
    return c


def energies(s, f):
    """orbital energies

    :param s: basis function overlap matrix
    :type s: numpy.ndarray
    :param f: fock matrix
    :type f: numpy.ndarray

    :rtype: numpy.ndarray
    """
    e, _ = spla.eigh(f, b=s)
    return e

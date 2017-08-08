import numpy
import scipy.linalg as spla


def density(c_o):
    """hf density matrix

    :param c_o: occupied orbital coefficients
    :type c_o: numpy.ndarray

    :rtype: numpy.ndarray
    """
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


def orbital_energies(s, f):
    """orbital energies

    :param s: basis function overlap matrix
    :type s: numpy.ndarray
    :param f: fock matrix
    :type f: numpy.ndarray

    :rtype: numpy.ndarray
    """
    e, _ = spla.eigh(f, b=s)
    return e

import numpy
import itertools as it

from . import elements


# Public
def energy(labels: tuple, coords: numpy.ndarray) -> float:
    """the coulomb energy of a system of nuclei

    :param labels: atomic symbols
    :type labels: tuple
    :param coords: coordinates
    :type coords: numpy.ndarray

    :return: the energy
    :rtype: float
    """
    charges = tuple(map(elements.charge, labels))
    return _inverse_law(weights=charges, coords=coords)


def electric_dipole(labels: tuple, coords: numpy.ndarray) -> numpy.ndarray:
    """the electric dipole moment of a system of nuclei

    :param labels: atomic symbols
    :type labels: tuple
    :param coords: coordinates
    :type coords: numpy.ndarray

    :return: the dipole vector
    :rtype: numpy.ndarray
    """
    charges = tuple(map(elements.charge, labels))
    return _moment(weights=charges, coords=coords)


# Private
def _inverse_law(weights: tuple, coords: numpy.ndarray) -> float:
    """the weighted sum of inverse distances for a system of nuclei

    :param weights: weights
    :type weights: tuple
    :param coords: coordinates
    :type coords: numpy.ndarray

    :rtype: float
    """
    w_pairs = it.combinations(weights, r=2)
    r_pairs = it.combinations(coords, r=2)
    return sum(w1 * w2 / numpy.linalg.norm(numpy.subtract(r1, r2))
               for (w1, w2), (r1, r2) in zip(w_pairs, r_pairs))


def _moment(weights: tuple, coords: numpy.ndarray) -> numpy.ndarray:
    """the weighted moment of a system of nuclei

    :param weights: weights
    :type weights: tuple
    :param coords: coordinates
    :type coords: numpy.ndarray

    :return: the moment vector
    :rtype: numpy.ndarray
    """
    return numpy.array(numpy.dot(weights, coords))

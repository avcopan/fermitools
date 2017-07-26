import numpy


def projection(s, d):
    """non-orthogonal projection onto an orbital space

    :param s: basis function overlap matrix
    :type s: numpy.ndarray
    :param d: orbital-space density matrix
    :type d: numpy.ndarray

    :rtype: numpy.ndarray
    """
    return numpy.dot(d, s)


def complementary_projection(s, d):
    """non-orthogonal projection onto the complement of an orbital space

    :param s: basis function overlap matrix
    :type s: numpy.ndarray
    :param d: orbital-space density matrix
    :type d: numpy.ndarray

    :rtype: numpy.ndarray
    """
    p = projection(s, d)
    return numpy.eye(*p.shape) - p

import numpy


def projection(s, d):
    return numpy.dot(d, s)


def complementary_projection(s, d):
    p = projection(s, d)
    return numpy.eye(*p.shape) - p

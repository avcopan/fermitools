"""second-order Møller-Plesset perturbation theory"""
import numpy


def doubles_amplitudes(goovv, e2):
    return numpy.divide(goovv, e2)

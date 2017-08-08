import numpy


def doubles_correlation_energy(aagoovv, abgoovv, bbgoovv, aat2, abt2, bbt2):
    aau2 = numpy.array(aat2) - numpy.transpose(aat2, (0, 1, 3, 2))
    bbu2 = numpy.array(bbt2) - numpy.transpose(bbt2, (0, 1, 3, 2))
    return (numpy.vdot(aagoovv, aau2) / 2. +
            numpy.vdot(abgoovv, abt2) +
            numpy.vdot(bbgoovv, bbu2) / 2.)

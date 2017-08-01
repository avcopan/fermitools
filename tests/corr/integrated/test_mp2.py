import fermitools.corr.integrated.mp2 as mp2

import os
import numpy
from numpy.testing import assert_almost_equal

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
NA = numpy.load(os.path.join(data_path, 'na.npy'))
NB = numpy.load(os.path.join(data_path, 'nb.npy'))
AAG = numpy.load(os.path.join(data_path, 'aag.npy'))
ABG = numpy.load(os.path.join(data_path, 'abg.npy'))
BBG = numpy.load(os.path.join(data_path, 'bbg.npy'))
AE = numpy.load(os.path.join(data_path, 'ae.npy'))
BE = numpy.load(os.path.join(data_path, 'be.npy'))
CORR_ENERGY = numpy.load(os.path.join(data_path, 'mp2/corr_energy.npy'))


def test__correlation_energy():
    corr_energy = mp2.correlation_energy(
            na=NA, nb=NB, aag=AAG, abg=ABG, bbg=BBG, ae=AE, be=BE)
    assert_almost_equal(corr_energy, CORR_ENERGY, decimal=10)

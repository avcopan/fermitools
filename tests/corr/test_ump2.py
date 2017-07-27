import fermitools.corr.ump2 as ump2

import os
import numpy
from numpy.testing import assert_almost_equal

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
NA = numpy.load(os.path.join(data_path, 'ump2/na.npy'))
NB = numpy.load(os.path.join(data_path, 'ump2/nb.npy'))
AAG = numpy.load(os.path.join(data_path, 'ump2/aag.npy'))
ABG = numpy.load(os.path.join(data_path, 'ump2/abg.npy'))
BBG = numpy.load(os.path.join(data_path, 'ump2/bbg.npy'))
AE = numpy.load(os.path.join(data_path, 'ump2/ae.npy'))
BE = numpy.load(os.path.join(data_path, 'ump2/be.npy'))
CORR_ENERGY = numpy.load(os.path.join(data_path, 'ump2/corr_energy.npy'))


def test__correlation_energy():
    corr_energy = ump2.correlation_energy(
            na=NA, nb=NB, aag=AAG, abg=ABG, bbg=BBG, ae=AE, be=BE)
    assert_almost_equal(corr_energy, CORR_ENERGY, decimal=10)

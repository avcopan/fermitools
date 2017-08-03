import fermitools.hf.spinorb.hf as spohf

import os
import numpy
from numpy.testing import assert_almost_equal

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
H = numpy.load(os.path.join(data_path, 'h.npy'))
G = numpy.load(os.path.join(data_path, 'g.npy'))
D = numpy.load(os.path.join(data_path, 'd.npy'))
F = numpy.load(os.path.join(data_path, 'f.npy'))
ENERGY = numpy.load(os.path.join(data_path, 'energy.npy'))


def test__fock():
    f = spohf.fock(h=H, g=G, d=D)
    assert_almost_equal(f, F, decimal=10)


def test__energy():
    energy = spohf.energy(h=H, f=F, d=D)
    assert_almost_equal(energy, ENERGY, decimal=10)

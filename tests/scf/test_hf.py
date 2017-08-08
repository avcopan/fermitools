import fermitools.scf.hf as hf

import os
import numpy
from numpy.testing import assert_almost_equal

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
H = numpy.load(os.path.join(data_path, 'hf/h.npy'))
G = numpy.load(os.path.join(data_path, 'hf/g.npy'))
D = numpy.load(os.path.join(data_path, 'hf/d.npy'))
F = numpy.load(os.path.join(data_path, 'hf/f.npy'))
ENERGY = numpy.load(os.path.join(data_path, 'hf/energy.npy'))


def test__fock():
    f = hf.fock(h=H, g=G, d=D)
    assert_almost_equal(f, F, decimal=10)


def test__energy():
    energy = hf.energy(h=H, f=F, d=D)
    assert_almost_equal(energy, ENERGY, decimal=10)

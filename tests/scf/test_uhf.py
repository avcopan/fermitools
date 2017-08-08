import fermitools.scf.uhf as uhf

import os
import numpy
from numpy.testing import assert_almost_equal

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
H = numpy.load(os.path.join(data_path, 'h.npy'))
G = numpy.load(os.path.join(data_path, 'g.npy'))
AD = numpy.load(os.path.join(data_path, 'uhf/ad.npy'))
BD = numpy.load(os.path.join(data_path, 'uhf/bd.npy'))
AF = numpy.load(os.path.join(data_path, 'uhf/af.npy'))
BF = numpy.load(os.path.join(data_path, 'uhf/bf.npy'))
ENERGY = numpy.load(os.path.join(data_path, 'uhf/energy.npy'))


def test__fock():
    af, bf = uhf.fock(h=H, g=G, ad=AD, bd=BD)
    assert_almost_equal(af, AF, decimal=10)
    assert_almost_equal(bf, BF, decimal=10)


def test__energy():
    energy = uhf.energy(h=H, af=AF, bf=BF, ad=AD, bd=BD)
    assert_almost_equal(energy, ENERGY, decimal=10)

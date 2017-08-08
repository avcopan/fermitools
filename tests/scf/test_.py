import fermitools.scf as scf

import os
import numpy
from numpy.testing import assert_almost_equal

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
C_O = numpy.load(os.path.join(data_path, 'rhf/c_o.npy'))
C = numpy.load(os.path.join(data_path, 'rhf/c.npy'))
D = numpy.load(os.path.join(data_path, 'rhf/d.npy'))
S = numpy.load(os.path.join(data_path, 's.npy'))
F = numpy.load(os.path.join(data_path, 'rhf/f.npy'))
E = numpy.load(os.path.join(data_path, 'rhf/e.npy'))


def test__density():
    d = scf.density(c_o=C_O)
    assert_almost_equal(d, D, decimal=10)


def test__coefficients():
    c = scf.coefficients(s=S, f=F)
    assert_almost_equal(numpy.abs(c), numpy.abs(C), decimal=10)


def test__orbital_energies():
    e = scf.orbital_energies(s=S, f=F)
    assert_almost_equal(e, E, decimal=10)


if __name__ == '__main__':
    test__density()

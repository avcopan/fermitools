import simplehf.hf.rohf as rohf

import os
import numpy
from numpy.testing import assert_almost_equal

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
S = numpy.load(os.path.join(data_path, 's.npy'))
AF = numpy.load(os.path.join(data_path, 'rohf/af.npy'))
BF = numpy.load(os.path.join(data_path, 'rohf/bf.npy'))
AD = numpy.load(os.path.join(data_path, 'rohf/ad.npy'))
BD = numpy.load(os.path.join(data_path, 'rohf/bd.npy'))
F = numpy.load(os.path.join(data_path, 'rohf/f.npy'))


def test__effective_fock():
    f = rohf.effective_fock(s=S, af=AF, bf=BF, ad=AD, bd=BD)
    assert_almost_equal(f, F, decimal=10)

import fermitools.oo.odc12 as odc12

import os
import numpy
from numpy.testing import assert_almost_equal

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

NITER = 200
RTHRESH = 1e-13
H_AO = numpy.load(os.path.join(data_path, 'h_ao.npy'))
R_AO = numpy.load(os.path.join(data_path, 'r_ao.npy'))
ACO_GUESS = numpy.load(os.path.join(data_path, 'aco_guess.npy'))
ACV_GUESS = numpy.load(os.path.join(data_path, 'acv_guess.npy'))
BCO_GUESS = numpy.load(os.path.join(data_path, 'bco_guess.npy'))
BCV_GUESS = numpy.load(os.path.join(data_path, 'bcv_guess.npy'))
T2_GUESS = numpy.load(os.path.join(data_path, 't2_guess.npy'))
ACO = numpy.load(os.path.join(data_path, 'odc12/aco.npy'))
ACV = numpy.load(os.path.join(data_path, 'odc12/acv.npy'))
BCO = numpy.load(os.path.join(data_path, 'odc12/bco.npy'))
BCV = numpy.load(os.path.join(data_path, 'odc12/bcv.npy'))
T2 = numpy.load(os.path.join(data_path, 'odc12/t2.npy'))
EN_ELEC = numpy.load(os.path.join(data_path, 'odc12/en_elec.npy'))


def test__solve():
    # test approximate guess
    en_elec, (aco, bco), (acv, bcv), t2, info = odc12.solve(
            H_AO, R_AO, (ACO_GUESS, BCO_GUESS), (ACV_GUESS, BCV_GUESS),
            T2_GUESS, NITER, RTHRESH, True)
    assert_almost_equal(en_elec, EN_ELEC, decimal=10)
    assert_almost_equal(aco, ACO, decimal=10)
    assert_almost_equal(acv, ACV, decimal=10)
    assert_almost_equal(bco, BCO, decimal=10)
    assert_almost_equal(bcv, BCV, decimal=10)
    assert_almost_equal(t2, T2, decimal=10)
    assert info['niter'] < 100
    assert info['r1max'] < RTHRESH
    assert info['r2max'] < RTHRESH
    # test perfect guess
    en_elec, (aco, bco), (acv, bcv), t2, info = odc12.solve(
            H_AO, R_AO, (ACO, BCO), (ACV, BCV), T2, NITER, RTHRESH, True)
    assert_almost_equal(en_elec, EN_ELEC, decimal=10)
    assert_almost_equal(aco, ACO, decimal=10)
    assert_almost_equal(acv, ACV, decimal=10)
    assert_almost_equal(bco, BCO, decimal=10)
    assert_almost_equal(bcv, BCV, decimal=10)
    assert_almost_equal(t2, T2, decimal=8)
    assert info['niter'] == 1
    assert info['r1max'] < RTHRESH
    assert info['r2max'] < RTHRESH


if __name__ == '__main__':
    test__solve()

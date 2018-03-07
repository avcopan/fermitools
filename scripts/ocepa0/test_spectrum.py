import os
import numpy
from numpy.testing import assert_almost_equal

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
W = numpy.load(os.path.join(data_path, 'neutral/w.npy'))


def test__main():
    import drivers.ocepa0
    import interfaces.psi4 as interface

    nroot = 7
    labels = ('O', 'H', 'H')
    coords = ((0.000000000000,  0.000000000000, -0.143225816552),
              (0.000000000000,  1.638036840407,  1.136548822547),
              (0.000000000000, -1.638036840407,  1.136548822547))

    w, info = drivers.ocepa0.spectrum(
            labels=labels,
            coords=coords,
            charge=0,
            spin=0,
            basis='sto-3g',
            angstrom=False,
            nroot=nroot,
            nguess=1,             # number of guess vectors per root
            nsvec=3,              # max number of sigma vectors per sub-iter
            nvec=100,             # max number of subspace vectors per root
            niter=50,             # number of iterations
            rthresh=1e-6,         # convergence threshold
            guess_random=True,    # use a random guess?
            oo_niter=200,         # number of iterations for ground state
            oo_rthresh=1e-10,     # convergence threshold for ground state
            diis_start=3,         # when to start DIIS extrapolations
            diis_nvec=20,         # maximum number of DIIS vectors
            disk=True,            #
            interface=interface)  # interface for computing integrals

    assert_almost_equal(w[:nroot], W[:nroot], decimal=10)


if __name__ == '__main__':
    test__main()

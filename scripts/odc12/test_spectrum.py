import os
import numpy
from numpy.testing import assert_almost_equal

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
W = numpy.load(os.path.join(data_path, 'neutral/w.npy'))


def test__main():
    import drivers.odc12
    import interfaces.psi4 as interface

    nroot = 10
    labels = ('O', 'H', 'H')
    coords = ((0.000000000000,  0.000000000000, -0.143225816552),
              (0.000000000000,  1.638036840407,  1.136548822547),
              (0.000000000000, -1.638036840407,  1.136548822547))

    w, info = drivers.odc12.spectrum(
            labels=labels,
            coords=coords,
            charge=0,
            spin=0,
            basis='sto-3g',
            angstrom=False,
            nroot=nroot,
            nconv=nroot,          # number of roots to converge
            nguess=5*nroot,       # number of guess vectors
            maxdim=7*nroot,      # max number of subspace vectors
            maxiter=100,
            rthresh=1e-9,
            oo_niter=200,         # number of iterations for ground state
            oo_rthresh=1e-10,     # convergence threshold for ground state
            diis_start=3,         # when to start DIIS extrapolations
            diis_nvec=20,         # maximum number of DIIS vectors
            disk=False,           #
            interface=interface)  # interface for computing integrals

    print(W[:nroot])
    assert_almost_equal(w[:nroot], W[:nroot], decimal=11)


if __name__ == '__main__':
    test__main()

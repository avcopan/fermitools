import drivers.ocepa0

import interfaces.psi4 as interface
# import interfaces.pyscf as interface

LABELS = ('N', 'N')
COORDS = ((0., 0., 0.), (0., 0., 1.5))


w, info = drivers.ocepa0.spectrum(
        labels=LABELS,
        coords=COORDS,
        charge=0,
        spin=0,
        basis='cc-pvdz',
        angstrom=True,
        nroot=20,
        nguess=12,              # number of guess vectors per root
        nsvec=10,               # max number of sigma vectors per sub-iteration
        nvec=100,               # max number of subspace vectors per root
        niter=50,               # number of iterations
        rthresh=1e-5,           # convergence threshold
        guess_random=False,
        oo_niter=200,           # number of iterations for ground state
        oo_rthresh=1e-8,        # convergence threshold for ground state
        diis_start=3,           # when to start DIIS extrapolations
        diis_nvec=20,           # maximum number of DIIS vectors
        disk=True,              #
        interface=interface)    # interface for computing integrals

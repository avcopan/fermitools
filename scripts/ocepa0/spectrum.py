import drivers.ocepa0

import interfaces.psi4 as interface
# import interfaces.pyscf as interface

LABELS = ('O', 'H', 'H')
COORDS = ((0.0000000000,  0.0000000000, -0.0678898741),
          (0.0000000000, -0.7507081111,  0.5387307840),
          (0.0000000000,  0.7507081111,  0.5387307840))


w, info = drivers.ocepa0.spectrum(
        labels=LABELS,
        coords=COORDS,
        charge=0,
        spin=0,
        basis='cc-pvdz',
        angstrom=True,
        nroot=20,
        nguess=12,              # number of guess vectors per root
        nsvec=2,                # max vectors per root per sub-iteration
        nvec=100,               # max number of subspace vectors per root
        niter=50,               # number of iterations
        rthresh=1e-5,           # convergence threshold
        guess_random=False,     # use a random guess?
        oo_niter=200,           # number of iterations for ground state
        oo_rthresh=1e-8,        # convergence threshold for ground state
        diis_start=3,           # when to start DIIS extrapolations
        diis_nvec=20,           # maximum number of DIIS vectors
        interface=interface)    # interface for computing integrals

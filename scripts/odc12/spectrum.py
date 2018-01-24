import drivers.odc12

import interfaces.psi4 as interface
# import interfaces.pyscf as interface

LABELS = ('O', 'H', 'H')
COORDS = ((0.0000000000,  0.0000000000, -0.0678898741),
          (0.0000000000, -0.7507081111,  0.5387307840),
          (0.0000000000,  0.7507081111,  0.5387307840))


w, x, mu_trans, info, oo_info = drivers.odc12.spectrum(
        labels=LABELS,
        coords=COORDS,
        charge=0,
        spin=0,
        basis='cc-pvdz',
        angstrom=True,
        nroot=20,
        nguess=12,              # number of guess vectors per root
        nvec=100,               # max number of subspace vectors per root
        niter=50,               # number of iterations
        rthresh=1e-5,           # convergence threshold
        guess_random=False,     # use a random guess?
        oo_niter=200,           # number of iterations for ground state
        oo_rthresh=1e-8,        # convergence threshold for ground state
        interface=interface)    # interface for computing integrals

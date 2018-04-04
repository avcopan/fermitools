import drivers.odc12

import interfaces.psi4 as interface
# import interfaces.pyscf as interface

LABELS = ('N', 'N')
COORDS = ((0., 0., 0.), (0., 0., 1.5))


w, info = drivers.odc12.spectrum(
        labels=LABELS,
        coords=COORDS,
        charge=0,
        spin=0,
        basis='3-21g',
        angstrom=True,
        nroot=7,
        nconv=7,                # number of roots to converge
        nguess=2*7,             # number of guess vectors
        maxdim=40*7,            # max number of subspace vectors
        maxiter=100,
        rthresh=1e-5,
        oo_maxiter=200,           # number of iterations for ground state
        oo_rthresh=1e-8,        # convergence threshold for ground state
        diis_start=3,           # when to start DIIS extrapolations
        diis_nvec=20,           # maximum number of DIIS vectors
        disk=False,
        interface=interface)    # interface for computing integrals

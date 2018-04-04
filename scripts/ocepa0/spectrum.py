import drivers

import interfaces.psi4 as interface
# import interfaces.pyscf as interface

LABELS = ('H', 'F')
COORDS = ((0., 0., 0.), (0., 0., 1.2))


drivers.spectrum(
        method='ocepa0',
        labels=LABELS,
        coords=COORDS,
        charge=0,
        spin=0,
        basis='3-21g',
        angstrom=True,
        nroot=12,
        nconv=10,
        nguess=10*5,
        maxdim=10*10,
        maxiter=50,
        rthresh=1e-5,
        oo_maxiter=200,         # number of iterations for ground state
        oo_rthresh=1e-8,        # convergence threshold for ground state
        diis_start=3,           # when to start DIIS extrapolations
        diis_nvec=20,           # maximum number of DIIS vectors
        disk=True,
        blsize=6,
        interface=interface)    # interface for computing integrals

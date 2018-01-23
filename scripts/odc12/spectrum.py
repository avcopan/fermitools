import drivers.odc12

import interfaces.psi4 as interface
# import interfaces.pyscf as interface

LABELS = ('O', 'H', 'H')
COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
          (0.000000000000,  1.638036840407,  1.136548822547),
          (0.000000000000, -1.638036840407,  1.136548822547))

w, x, info, oo_info = drivers.odc12.spectrum(
        labels=LABELS,
        coords=COORDS,
        charge=0,
        spin=0,
        basis='cc-pvdz',
        angstrom=False,
        nroot=7,
        nguess=10,              # number of guess vectors per root
        nvec=100,               # max number of subspace vectors per root
        niter=50,               # number of iterations
        rthresh=1e-6,           # convergence threshold
        oo_niter=200,           # number of iterations for ground state
        oo_rthresh=1e-10,       # convergence threshold for ground state
        interface=interface)    # interface for computing integrals

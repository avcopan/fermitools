import drivers.cepa0

import interfaces.psi4 as interface

LABELS = ('O', 'H', 'H')
COORDS = ((0.0000000000,  0.0000000000, -0.0678898741),
          (0.0000000000, -0.7507081111,  0.5387307840),
          (0.0000000000,  0.7507081111,  0.5387307840))


en_corr, t2, info = drivers.cepa0.energy(
        labels=LABELS,
        coords=COORDS,
        charge=1,
        spin=1,
        basis='cc-pvdz',
        angstrom=True,
        niter=200,            # number of iterations for ground state
        rthresh=1e-12,        # convergence threshold for ground state
        interface=interface)  # interface for computing integrals

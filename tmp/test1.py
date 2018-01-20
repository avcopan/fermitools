import numpy

import fermitools
import interfaces.psi4 as interface
# import interfaces.pyscf as interface

CHARGE = +0
SPIN = 0
BASIS = 'sto-3g'
LABELS = ('O', 'H', 'H')
COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
          (0.000000000000,  1.638036840407,  1.136548822547),
          (0.000000000000, -1.638036840407,  1.136548822547))

# Ground state options
OO_NITER = 200        # number of iterations
OO_RTHRESH = 1e-10  # convergence threshold

# Excited state options
LR_NROOT = 7        # number of roots
LR_NGUESS = 2       # number of guess vectors per root
LR_NVEC = 20        # number of subspace vectors per root
LR_NITER = 200      # number of iterations
LR_RTHRESH = 1e-11  # convergence threshold


def main():
    # Spaces
    na = fermitools.chem.elec.count_alpha(LABELS, CHARGE, SPIN)
    nb = fermitools.chem.elec.count_beta(LABELS, CHARGE, SPIN)

    # Integrals
    h_ao = interface.integrals.core_hamiltonian(BASIS, LABELS, COORDS)
    r_ao = interface.integrals.repulsion(BASIS, LABELS, COORDS)

    # Mean-field guess orbitals
    c_guess = interface.hf.unrestricted_orbitals(
            BASIS, LABELS, COORDS, CHARGE, SPIN)

    # nbf = interface.integrals.nbf(BASIS, LABELS)
    # nso = 2 * nbf
    # no = na + nb
    # nv = nso - no
    # t2_guess = numpy.zeros((no, no, nv, nv))
    t2_guess = numpy.load('tmp/t2.npy')

    # Solve ground state
    en_elec, c, t2, info = fermitools.oo.odc12.solve_new(
            na=na, nb=nb, h_ao=h_ao, r_ao=r_ao,
            c_guess=c_guess, t2_guess=t2_guess,
            niter=OO_NITER, r_thresh=OO_RTHRESH)
    en_nuc = fermitools.chem.nuc.energy(labels=LABELS, coords=COORDS)
    en_tot = en_elec + en_nuc
    print("\nGround state energy:")
    print('{:20.15f}'.format(en_tot))


if __name__ == '__main__':
    main()

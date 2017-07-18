import pyscf

import numpy


# Public
def unrestricted_orbitals(basis: str, labels: tuple, coords: numpy.ndarray,
                          charge: int=0, spin: int=0, niter: int=100,
                          e_thresh: float=1e-12,
                          r_thresh: float=1e-9) -> numpy.ndarray:
    """urestricted alpha and beta Hartree-Fock orbital coefficients

    :param basis: basis set name
    :type basis: str
    :param labels: atomic symbols labeling the nuclei
    :type labels: tuple
    :param coords: nuclear coordinates in Bohr
    :type coords: numpy.ndarray
    :param charge: total molecular charge
    :type charge: int
    :param spin: number of unpaired electrons
    :type spin: int
    :param niter: maximum number of iterations
    :type niter: int
    :param e_thresh: energy convergence threshold
    :type e_thresh: float
    :param r_thresh: residual convergence threshold
    :type r_thresh: float

    :return: an array of two square matrices
    :rtype: numpy.ndarray
    """
    molecule = pyscf.gto.Mole(atom=zip(labels, coords),
                              unit="bohr",
                              basis=basis,
                              charge=charge,
                              spin=spin)
    molecule.build()
    uhf = pyscf.scf.UHF(molecule)
    uhf.max_cycle = niter
    uhf.conv_tol = e_thresh
    uhf.conv_tol_grad = r_thresh
    uhf.kernel()
    return uhf.mo_coeff


def restricted_orbitals(basis: str, labels: tuple, coords: numpy.ndarray,
                        charge: int=0, spin: int=0, niter: int=100,
                        e_thresh: float=1e-12,
                        r_thresh: float=1e-9) -> numpy.ndarray:
    """restricted Hartree-Fock orbital coefficients

    :param basis: basis set name
    :type basis: str
    :param labels: atomic symbols labeling the nuclei
    :type labels: tuple
    :param coords: nuclear coordinates in Bohr
    :type coords: numpy.ndarray
    :param charge: total molecular charge
    :type charge: int
    :param spin: number of unpaired electrons
    :type spin: int
    :param niter: maximum number of iterations
    :type niter: int
    :param e_thresh: energy convergence threshold
    :type e_thresh: float
    :param r_thresh: residual convergence threshold
    :type r_thresh: float

    :return: a square matrix
    :rtype: numpy.ndarray
    """
    molecule = pyscf.gto.Mole(atom=zip(labels, coords),
                              unit="bohr",
                              basis=basis,
                              charge=charge,
                              spin=spin)
    molecule.build()
    rhf = pyscf.scf.RHF(molecule)
    rhf.max_cycle = niter
    rhf.conv_tol = e_thresh
    rhf.conv_tol_grad = r_thresh
    rhf.kernel()
    return rhf.mo_coeff


# Testing
def _main():
    BASIS = 'STO-3G'
    LABELS = ('O', 'H', 'H')
    COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
              (0.000000000000,  1.638036840407,  1.136548822547),
              (0.000000000000, -1.638036840407,  1.136548822547))
    CHARGE = 0
    SPIN = 0
    ac, bc = unrestricted_orbitals(basis=BASIS, labels=LABELS,
                                   coords=COORDS, charge=CHARGE,
                                   spin=SPIN)
    print(numpy.linalg.norm(ac))
    print(numpy.linalg.norm(bc))


if __name__ == "__main__":
    _main()

import psi4
import numpy

from .util import psi4_molecule

import multiprocessing

psi4.set_num_threads(multiprocessing.cpu_count())


# Public
def unrestricted_orbitals(basis, labels, coords, charge=0, spin=0, niter=100,
                          e_thresh=1e-12, r_thresh=1e-9, guess='gwh'):
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
    :param guess: hartree-fock starting guess
    :type guess: str

    :return: an array of two square matrices
    :rtype: numpy.ndarray
    """
    psi4.set_options({'e_convergence': e_thresh, 'd_convergence': r_thresh,
                      'maxiter': niter, 'guess': guess, 'reference': 'RHF'})
    mol = psi4_molecule(labels=labels, coords=coords, charge=charge, spin=spin)
    wfn = psi4.core.Wavefunction.build(mol, basis)
    sf, _ = psi4.driver.dft_funcs.build_superfunctional("HF", False)
    hf = psi4.core.UHF(wfn, sf)
    hf.compute_energy()
    ac = numpy.array(hf.Ca())
    bc = numpy.array(hf.Cb())
    return numpy.array([ac, bc])


def restricted_orbitals(basis, labels, coords, charge=0, spin=0, niter=100,
                        e_thresh=1e-12, r_thresh=1e-9, guess='gwh'):
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
    :param guess: hartree-fock starting guess
    :type guess: str

    :return: a square matrix
    :rtype: numpy.ndarray
    """
    psi4.set_options({'e_convergence': e_thresh, 'd_convergence': r_thresh,
                      'maxiter': niter, 'guess': guess, 'reference': 'RHF'})
    mol = psi4_molecule(labels=labels, coords=coords, charge=charge, spin=spin)
    wfn = psi4.core.Wavefunction.build(mol, basis)
    sf, _ = psi4.driver.dft_funcs.build_superfunctional("HF", False)

    if spin is 0:
        psi4.set_options({'reference': 'RHF'})
        hf = psi4.core.RHF(wfn, sf)
    else:
        psi4.set_options({'reference': 'ROHF'})
        hf = psi4.core.ROHF(wfn, sf)

    hf.compute_energy()
    c = numpy.array(hf.Ca())
    return c


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

import psi4.core

import numpy


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
    mol = _psi4_molecule_object(basis=basis,
                                labels=labels,
                                coords=coords,
                                charge=charge,
                                spin=spin)
    wfn = psi4.core.Wavefunction.build(mol, basis)
    sf, _ = psi4.driver.dft_funcs.build_superfunctional("HF", False)
    psi4.core.set_global_option('guess', guess)
    psi4.core.set_global_option('e_convergence', e_thresh)
    psi4.core.set_global_option('d_convergence', r_thresh)
    psi4.core.set_global_option('maxiter', niter)
    psi4.core.set_global_option('reference', 'UHF')
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
    mol = _psi4_molecule_object(basis=basis,
                                labels=labels,
                                coords=coords,
                                charge=charge,
                                spin=spin)
    wfn = psi4.core.Wavefunction.build(mol, basis)
    sf, _ = psi4.driver.dft_funcs.build_superfunctional("HF", False)
    psi4.core.set_global_option('guess', guess)
    psi4.core.set_global_option('e_convergence', e_thresh)
    psi4.core.set_global_option('d_convergence', r_thresh)
    psi4.core.set_global_option('maxiter', niter)

    if spin is 0:
        psi4.core.set_global_option('reference', 'RHF')
        hf = psi4.core.RHF(wfn, sf)
    else:
        psi4.core.set_global_option('reference', 'ROHF')
        hf = psi4.core.ROHF(wfn, sf)

    hf.compute_energy()
    c = numpy.array(hf.Ca())
    return c


# Private
def _coordinate_string(labels, coords):
    """coordinate string

    :param labels: atomic symbols labeling the nuclei
    :type labels: tuple
    :param coords: nuclear coordinates in Bohr
    :type coords: numpy.ndarray

    :rtype: str
    """
    coord_line_template = "{:2s} {: >17.12f} {: >17.12f} {: >17.12f}"
    coord_str = "\n".join(coord_line_template.format(label, *coord)
                          for label, coord in zip(labels, coords))
    coord_str += "\nunits bohr"
    return coord_str


def _psi4_molecule_object(basis, labels, coords, charge, spin):
    """build a Psi4 Molecule object

    :param basis: basis set name
    :type basis: str
    :param labels: atomic symbols labeling the nuclei
    :type labels: tuple
    :param coords: nuclear coordinates in Bohr
    :type coords: numpy.ndarray

    :rtype: psi4.core.Molecule
    """
    coord_str = _coordinate_string(labels=labels, coords=coords)
    mol = psi4.core.Molecule.create_molecule_from_string(coord_str)
    mol.set_molecular_charge(charge)
    mol.set_multiplicity(spin + 1)
    mol.reset_point_group('C1')
    mol.update_geometry()

    return mol


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

import psi4

import multiprocessing

psi4.set_num_threads(multiprocessing.cpu_count())


def psi4_basis(basis, labels, coords):
    """
    :param basis: basis set name
    :type basis: str
    :param labels: atomic symbols labeling the nuclei
    :type labels: tuple
    :param coords: nuclear coordinates in Bohr
    :type coords: numpy.ndarray
    """
    psi4.core.clean()
    mol = psi4_molecule(labels=labels, coords=coords)
    bs = psi4.core.BasisSet.build(mol, '', basis)
    return bs


def psi4_molecule(labels, coords, charge=0, spin=0):
    """
    :param labels: atomic symbols labeling the nuclei
    :type labels: tuple
    :param coords: nuclear coordinates in Bohr
    :type coords: numpy.ndarray
    :param charge: total molecular charge
    :type charge: int
    :param spin: number of unpaired electrons
    :type spin: int
    :rtype: psi4.core.Molecule
    """
    xyz = coordinate_string(labels=labels, coords=coords)
    mol = psi4.core.Molecule.create_molecule_from_string(xyz)
    mol.set_molecular_charge(charge)
    mol.set_multiplicity(spin + 1)
    mol.reset_point_group('C1')
    mol.update_geometry()
    return mol


def coordinate_string(labels, coords):
    """coordinate string
    :param labels: atomic symbols labeling the nuclei
    :type labels: tuple
    :param coords: nuclear coordinates in Bohr
    :type coords: numpy.ndarray
    :returns: the molecular geometry
    :rtype: str
    """
    line = '{:2s} {: >17.12f} {: >17.12f} {: >17.12f}'
    xyz = '\n'.join(line.format(l, *c) for l, c in zip(labels, coords))
    return 'units bohr\n{:s}'.format(xyz)

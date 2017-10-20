import psi4.core

import numpy


# Public
def nbf(basis, labels):
    coords = numpy.random.rand(3, len(labels))
    mints = _psi4_mints_object(basis=basis, labels=labels, coords=coords)
    return int(mints.nbf())


def overlap(basis, labels, coords):
    """overlap integrals

    :param basis: basis set name
    :type basis: str
    :param labels: atomic symbols labeling the nuclei
    :type labels: tuple
    :param coords: nuclear coordinates in Bohr
    :type coords: numpy.ndarray

    :return: a square matrix
    :rtype: numpy.ndarray
    """
    mints = _psi4_mints_object(basis=basis, labels=labels, coords=coords)
    return numpy.array(mints.ao_overlap())


def kinetic(basis, labels, coords):
    """kinetic energy integrals

    :param basis: basis set name
    :type basis: str
    :param labels: atomic symbols labeling the nuclei
    :type labels: tuple
    :param coords: nuclear coordinates in Bohr
    :type coords: numpy.ndarray

    :return: a square matrix
    :rtype: numpy.ndarray
    """
    mints = _psi4_mints_object(basis=basis, labels=labels, coords=coords)
    return numpy.array(mints.ao_kinetic())


def nuclear(basis, labels, coords):
    """nuclear attraction integrals

    :param basis: basis set name
    :type basis: str
    :param labels: atomic symbols labeling the nuclei
    :type labels: tuple
    :param coords: nuclear coordinates in Bohr
    :type coords: numpy.ndarray

    :return: a square matrix
    :rtype: numpy.ndarray
    """
    mints = _psi4_mints_object(basis=basis, labels=labels, coords=coords)
    return numpy.array(mints.ao_potential())


def core_hamiltonian(basis, labels, coords):
    """kinetic energy plus nuclear attraction integrals

    :param basis: basis set name
    :type basis: str
    :param labels: atomic symbols labeling the nuclei
    :type labels: tuple
    :param coords: nuclear coordinates in Bohr
    :type coords: numpy.ndarray

    :return: a square matrix
    :rtype: numpy.ndarray
    """
    t = kinetic(basis=basis, labels=labels, coords=coords)
    v = nuclear(basis=basis, labels=labels, coords=coords)
    return t + v


def dipole(basis, labels, coords):
    """electric dipole integrals

    :param basis: basis set name
    :type basis: str
    :param labels: atomic symbols labeling the nuclei
    :type labels: tuple
    :param coords: nuclear coordinates in Bohr
    :type coords: numpy.ndarray

    :return: an array of three square matrices
    :rtype: numpy.ndarray
    """
    mints = _psi4_mints_object(basis=basis, labels=labels, coords=coords)
    return numpy.array(tuple(map(numpy.array, mints.ao_dipole())))


def repulsion(basis, labels, coords):
    """electron-electron repulsion integrals in physicist's notation

    :param basis: basis set name
    :type basis: str
    :param labels: atomic symbols labeling the nuclei
    :type labels: tuple
    :param coords: nuclear coordinates in Bohr
    :type coords: numpy.ndarray

    :return: a four-index tensor of equal dimensions
    :rtype: numpy.ndarray
    """
    mints = _psi4_mints_object(basis=basis, labels=labels, coords=coords)
    return numpy.array(mints.ao_eri()).transpose((0, 2, 1, 3))


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


def _psi4_mints_object(basis, labels, coords):
    """build a Psi4 MintsHelper object

    :param basis: basis set name
    :type basis: str
    :param labels: atomic symbols labeling the nuclei
    :type labels: tuple
    :param coords: nuclear coordinates in Bohr
    :type coords: numpy.ndarray

    :rtype: psi4.core.MintsHelper
    """
    coord_str = _coordinate_string(labels=labels, coords=coords)
    mol = psi4.core.Molecule.create_molecule_from_string(coord_str)
    mol.reset_point_group("c1")
    mol.update_geometry()

    basis_obj, _ = psi4.core.BasisSet.build(mol, 'BASIS', basis)
    mints_obj = psi4.core.MintsHelper(basis_obj)

    return mints_obj

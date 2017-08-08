import pyscf

import numpy


# Public
def nbf(basis, labels):
    coords = ((0, 0, 0),) * len(labels)
    mol = _pyscf_molecule_object(basis=basis, labels=labels, coords=coords)
    return mol.nao_nr()


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
    mol = _pyscf_molecule_object(basis=basis, labels=labels, coords=coords)
    return mol.intor('cint1e_ovlp_sph')


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
    mol = _pyscf_molecule_object(basis=basis, labels=labels, coords=coords)
    return mol.intor('cint1e_kin_sph')


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
    mol = _pyscf_molecule_object(basis=basis, labels=labels, coords=coords)
    return mol.intor('cint1e_nuc_sph')


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
    mol = _pyscf_molecule_object(basis=basis, labels=labels, coords=coords)
    return -mol.intor('cint1e_r_sph', comp=3)


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
    mol = _pyscf_molecule_object(basis=basis, labels=labels, coords=coords)
    matrix = mol.intor('cint2e_sph')
    n = int(numpy.sqrt(matrix.shape[0]))
    return matrix.reshape((n, n, n, n)).transpose((0, 2, 1, 3))


# Private
def _pyscf_molecule_object(basis, labels, coords):
    """build a pyscf.gto.Mole object

    :param basis: basis set name
    :type basis: str
    :param labels: atomic symbols labeling the nuclei
    :type labels: tuple
    :param coords: nuclear coordinates in Bohr
    :type coords: numpy.ndarray

    :rtype: pyscf.gto.Mole
    """
    mol = pyscf.gto.Mole(atom=zip(labels, coords),
                         unit="bohr",
                         basis=basis)
    mol.build()
    return mol

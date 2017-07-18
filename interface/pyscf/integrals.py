import pyscf

import numpy


# Public
def overlap(basis: str, labels: tuple, coords: numpy.ndarray) -> numpy.ndarray:
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


def kinetic(basis: str, labels: tuple, coords: numpy.ndarray) -> numpy.ndarray:
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


def nuclear(basis: str, labels: tuple, coords: numpy.ndarray) -> numpy.ndarray:
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


def dipole(basis: str, labels: tuple, coords: numpy.ndarray) -> numpy.ndarray:
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


def repulsion(basis: str, labels: tuple,
              coords: numpy.ndarray) -> numpy.ndarray:
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
def _pyscf_molecule_object(basis: str, labels: tuple,
                           coords: numpy.ndarray) -> pyscf.gto.Mole:
    """build a pyscf.gto.Mole object

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

    :rtype: pyscf.gto.Mole
    """
    mol = pyscf.gto.Mole(atom=zip(labels, coords),
                         unit="bohr",
                         basis=basis)
    mol.build()
    return mol


# Testing
def _main():
    BASIS = 'STO-3G'
    LABELS = ('O', 'H', 'H')
    COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
              (0.000000000000,  1.638036840407,  1.136548822547),
              (0.000000000000, -1.638036840407,  1.136548822547))
    s = repulsion(basis=BASIS, labels=LABELS, coords=COORDS)
    print(numpy.linalg.norm(s))


if __name__ == "__main__":
    _main()

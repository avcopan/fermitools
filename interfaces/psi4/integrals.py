import psi4
import numpy
import scipy.linalg

from .util import psi4_basis

import multiprocessing

psi4.set_num_threads(multiprocessing.cpu_count())


# Public
def nbf(basis, labels):
    coords = numpy.random.rand(len(labels), 3)
    bs = psi4_basis(basis=basis, labels=labels, coords=coords)
    return int(bs.nbf())


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
    bs = psi4_basis(basis=basis, labels=labels, coords=coords)
    mh = psi4.core.MintsHelper(bs)
    return numpy.ascontiguousarray(mh.ao_overlap())


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
    bs = psi4_basis(basis=basis, labels=labels, coords=coords)
    mh = psi4.core.MintsHelper(bs)
    return numpy.ascontiguousarray(mh.ao_kinetic())


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
    bs = psi4_basis(basis=basis, labels=labels, coords=coords)
    mh = psi4.core.MintsHelper(bs)
    return numpy.ascontiguousarray(mh.ao_potential())


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


def coulomb_metric(basis, labels, coords):
    """coulomb metric integrals, (P|r12^-1|Q)

    :param basis: basis set name
    :type basis: str
    :param labels: atomic symbols labeling the nuclei
    :type labels: tuple
    :param coords: nuclear coordinates in Bohr
    :type coords: numpy.ndarray

    :return: a square matrix
    :rtype: numpy.ndarray
    """
    bs0 = psi4.core.BasisSet.zero_ao_basis_set()
    bs = psi4_basis(basis=basis, labels=labels, coords=coords)

    mh = psi4.core.MintsHelper(bs)
    cm = numpy.squeeze(mh.ao_eri(bs, bs0, bs, bs0))
    return numpy.ascontiguousarray(cm)


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
    bs = psi4_basis(basis=basis, labels=labels, coords=coords)
    mh = psi4.core.MintsHelper(bs)
    d = tuple(map(numpy.ascontiguousarray, mh.ao_dipole()))
    return numpy.ascontiguousarray(d)


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
    bs = psi4_basis(basis=basis, labels=labels, coords=coords)
    mh = psi4.core.MintsHelper(bs)
    r = numpy.array(mh.ao_eri()).transpose((0, 2, 1, 3))
    return numpy.ascontiguousarray(r)


def threecenter_repulsion(basis1, basis2, basis3, labels, coords):
    """three-center electron-electron repulsion integrals

    :param basis1: basis set name for first center (electron 1)
    :type basis1: str
    :param basis2: basis set name for second center (electron 1)
    :type basis2: str
    :param basis3: basis set name for third center (electron 2)
    :type basis3: str
    :param labels: atomic symbols labeling the nuclei
    :type labels: tuple
    :param coords: nuclear coordinates in Bohr
    :type coords: numpy.ndarray

    :return: three-index tensor
    :rtype: numpy.ndarray
    """
    bs0 = psi4.core.BasisSet.zero_ao_basis_set()
    bs1 = psi4_basis(basis=basis1, labels=labels, coords=coords)
    bs2 = psi4_basis(basis=basis2, labels=labels, coords=coords)
    bs3 = psi4_basis(basis=basis3, labels=labels, coords=coords)
    mh = psi4.core.MintsHelper(bs1)
    r3c = numpy.squeeze(mh.ao_eri(bs1, bs2, bs3, bs0))
    return numpy.ascontiguousarray(r3c)


def factorized_repulsion(basis, auxbasis, labels, coords):
    """electron-electron repulsion, factorized by resolution of the identity

    :param basis: basis set name
    :type basis: str
    :param auxbasis: auxiliary basis set name
    :type auxbasis: str
    :param labels: atomic symbols labeling the nuclei
    :type labels: tuple
    :param coords: nuclear coordinates in Bohr
    :type coords: numpy.ndarray

    :return: three-index tensor; last axis is the factorization index
    :rtype: numpy.ndarray
    """
    cm = coulomb_metric(basis=auxbasis, labels=labels, coords=coords)
    xy = scipy.linalg.cholesky(scipy.linalg.inv(cm))
    rijy = threecenter_repulsion(
            basis1=basis, basis2=basis, basis3=auxbasis, labels=labels,
            coords=coords)
    rijx = numpy.tensordot(rijy, xy, axes=(-1, -1))
    return numpy.ascontiguousarray(rijx)

from . import elements


def count(labels: tuple, charge: int=0) -> int:
    """the number of electrons in a chemical system

    :param labels: nuclear labels
    :type labels: tuple
    :param charge: total charge of the system
    :type charge: int

    :return: the number of electrons
    :rtype: int
    """
    nuc_charge = sum(map(elements.charge, labels))
    return nuc_charge - charge


def count_alpha(labels: tuple, charge: int=0, spin: int=0) -> int:
    """the number of alpha electrons in a high-spin chemical system

    :param labels: nuclear labels
    :type labels: tuple
    :param charge: total charge of the system
    :type charge: int
    :param spin: number of unpaired electrons
    :type spin: int

    :return: the number of alpha electrons
    :rtype: int
    """
    nelec = count(labels=labels, charge=charge)
    return (nelec - spin) // 2 + spin


def count_beta(labels: tuple, charge: int=0, spin: int=0) -> int:
    """the number of beta electrons in a high-spin chemical system

    :param labels: nuclear labels
    :type labels: tuple
    :param charge: total charge of the system
    :type charge: int
    :param spin: number of unpaired electrons
    :type spin: int

    :return: the number of beta electrons
    :rtype: int
    """
    nelec = count(labels=labels, charge=charge)
    return (nelec - spin) // 2

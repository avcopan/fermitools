import numpy.linalg as npla
from ._proj import projection, complementary_projection


def effective_fock(s, af, bf, ad, bd):
    """spin-restricted open-shell effective fock matrix

    :param s: basis function overlap matrix
    :type s: numpy.ndarray
    :param af: alpha fock matrix
    :type af: numpy.ndarray
    :param bf: beta fock matrix
    :type bf: numpy.ndarray
    :param ad: hartree-fock alpha density matrix
    :type ad: numpy.ndarray
    :param bd: hatree-fock beta density matrix
    :type bd: numpy.ndarray

    :rtype: numpy.ndarray
    """
    p_docc = projection(s, bd)
    p_socc = projection(s, ad - bd)
    p_uocc = complementary_projection(s, ad)
    p_iact = p_docc + p_uocc
    f_avg = (af + bf) / 2.
    f_eff = (npla.multi_dot([p_docc.T, bf, p_socc]) +
             npla.multi_dot([p_socc.T, bf, p_docc]) +
             npla.multi_dot([p_socc.T, af, p_uocc]) +
             npla.multi_dot([p_uocc.T, af, p_socc]) +
             npla.multi_dot([p_iact.T, f_avg, p_iact]) +
             npla.multi_dot([p_socc.T, f_avg, p_socc]))
    return f_eff

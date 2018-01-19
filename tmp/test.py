import scipy

import fermitools
import interfaces.psi4 as interface

CHARGE = +0
SPIN = 0
BASIS = 'sto-3g'
LABELS = ('O', 'H', 'H')
COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
          (0.000000000000,  1.638036840407,  1.136548822547),
          (0.000000000000, -1.638036840407,  1.136548822547))


def main():
    r_ao = interface.integrals.repulsion(BASIS, LABELS, COORDS)

    r_aso = fermitools.math.spinorb.expand(r_ao, brakets=((0, 2), (1, 3)))

    # Mean-field guess orbitals
    ac, bc = interface.hf.unrestricted_orbitals(
            BASIS, LABELS, COORDS, CHARGE, SPIN)
    c = scipy.linalg.block_diag(ac, bc)

    r = fermitools.math.transform(r_aso, c, c, c, c)
    print(r.shape)

    abc = (ac, bc)
    r_alt = fermitools.math.spinorb.transform(r_ao, (abc, abc, abc, abc),
                                              brakets=((0, 2), (1, 3)))
    print(r_alt.shape)

    x = r[:7, :7, :7, :7] - r_alt[:7, :7, :7, :7]
    print(x[0, 0].round(1))

    from numpy.testing import assert_almost_equal
    assert_almost_equal(r[:7, :7, :7, :7], r_alt[:7, :7, :7, :7])


if __name__ == '__main__':
    main()

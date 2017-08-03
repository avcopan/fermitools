import numpy
import fermitools
import fermitools.interface.pyscf as interface


def t2_amplitudes(goovv, eo, ev):
    t2 = goovv / fermitools
    t2 = goovv * fermitools.corr.resolvent((eo, eo), (ev, ev),)
    return t2


def correlation_energy(goovv, t2):
    return numpy.vdot(goovv, t2) / 4.


def main(basis, labels, coords, charge, spin):
    nbf = interface.integrals.nbf(basis, labels)
    h_ao = interface.integrals.core_hamiltonian(basis, labels, coords)
    r_ao = interface.integrals.repulsion(basis, labels, coords)

    h_aso = fermitools.math.spinorb.expand(h_ao, brakets=((0, 1),))
    r_aso = fermitools.math.spinorb.expand(r_ao, brakets=((0, 2), (1, 3)))
    g_aso = r_aso - numpy.transpose(r_aso, (0, 1, 3, 2))

    na = fermitools.chem.elec.count_alpha(labels, charge, spin)
    nb = fermitools.chem.elec.count_beta(labels, charge, spin)
    n = na + nb

    ac, bc = interface.hf.unrestricted_orbitals(basis, labels, coords)

    from scipy.linalg import block_diag
    from fermitools.math.spinorb import ab2ov

    c = fermitools.math.spinorb.sort(a=block_diag(ac, bc),
                                     order=ab2ov(nbf, na, nb),
                                     axes=(1,))

    d_aso = fermitools.hf.orb.density(n, c)
    f_aso = fermitools.hf.spohf.fock(h=h_aso, g=g_aso, d=d_aso)

    print(numpy.vdot(h_aso + f_aso, d_aso) / 2. +
          fermitools.chem.nuc.energy(labels, coords))

    f = fermitools.math.trans.transform(f_aso, {0: c, 1: c})
    g = fermitools.math.trans.transform(g_aso, {0: c, 1: c, 2: c, 3: c})

    e = numpy.diagonal(f)

    o = slice(None, n)
    v = slice(n, None)

    t2 = t2_amplitudes(g[o, o, v, v], e[o], e[v])

    corr_energy = correlation_energy(g[o, o, v, v], t2)

    print(corr_energy)


if __name__ == '__main__':
    CHARGE = +1
    SPIN = 1
    BASIS = 'STO-3G'
    LABELS = ('O', 'H', 'H')
    COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
              (0.000000000000,  1.638036840407,  1.136548822547),
              (0.000000000000, -1.638036840407,  1.136548822547))

    main(basis=BASIS, labels=LABELS, coords=COORDS, charge=CHARGE, spin=SPIN)

import simplehf
import simplehf.interface.pyscf as interface

CHARGE = +1
SPIN = 1
BASIS = 'STO-3G'
LABELS = ('O', 'H', 'H')
COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
          (0.000000000000,  1.638036840407,  1.136548822547),
          (0.000000000000, -1.638036840407,  1.136548822547))

h = interface.integrals.core_hamiltonian(BASIS, LABELS, COORDS)
g = interface.integrals.repulsion(BASIS, LABELS, COORDS)

na = simplehf.chem.elec.count_alpha(LABELS, CHARGE, SPIN)
nb = simplehf.chem.elec.count_beta(LABELS, CHARGE, SPIN)

ac, bc = interface.hf.unrestricted_orbitals(BASIS, LABELS, COORDS,
                                            CHARGE, SPIN)

ad = simplehf.hf.orb.density(na, ac)
bd = simplehf.hf.orb.density(nb, bc)

af, bf = simplehf.hf.uhf.fock(h, g, ad, bd)

energy = simplehf.hf.uhf.energy(h, af, bf, ad, bd)

nuc_energy = simplehf.chem.nuc.energy(LABELS, COORDS)

print(energy + nuc_energy)

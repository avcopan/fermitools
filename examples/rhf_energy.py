import simplehf
import simplehf.interface.pyscf as interface

CHARGE = 0
SPIN = 0
BASIS = 'STO-3G'
LABELS = ('O', 'H', 'H')
COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
          (0.000000000000,  1.638036840407,  1.136548822547),
          (0.000000000000, -1.638036840407,  1.136548822547))

s = interface.integrals.overlap(BASIS, LABELS, COORDS)
h = interface.integrals.core_hamiltonian(BASIS, LABELS, COORDS)
g = interface.integrals.repulsion(BASIS, LABELS, COORDS)

n = simplehf.chem.elec.count(LABELS, CHARGE) // 2

c = interface.hf.restricted_orbitals(BASIS, LABELS, COORDS)

d = simplehf.hf.orb.density(n, c)

f = simplehf.hf.rhf.fock(h, g, d)

energy = simplehf.hf.rhf.energy(h, f, d)

nuc_energy = simplehf.chem.nuc.energy(LABELS, COORDS)

print(energy + nuc_energy)

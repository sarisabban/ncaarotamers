#!/usr/bin/env python3
# Backbone-dependent rotamer library for a non-canonical amino acid,
# following the MakeRotLib protocol (Renfrew et al., PLoS ONE 2012).
# Rosetta's own make_rot_lib application is not distributed with PyRosetta,
# so the protocol is reimplemented here on the public PyRosetta API.
#   python3 NCAA_script.py ORN > ORN.json

import sys, json, math, itertools, pyrosetta
from pyrosetta import get_score_function, MoveMap
from pyrosetta.rosetta.core.chemical import ChemicalManager, VariantType
from pyrosetta.rosetta.core.conformation import ResidueFactory
from pyrosetta.rosetta.protocols.minimization_packing import MinMover

# base residue, phosphorylation patch, and the rotational symmetry of each
# chi in degrees (None = none, 120 = terminal PO3, 180 = aromatic ring flip)
RES = {'ORN': ('ORN', None,             [None, None, None]),
       'TPO': ('THR', 'PHOSPHORYLATION', [None, None, 120]),
       'PTR': ('TYR', 'PHOSPHORYLATION', [None, 180, None, 120])}
WELLS, KT, STEP = (60.0, 180.0, 300.0), 0.001987 * 300.0, 30.0

tricode = sys.argv[1]
base, variant, sym = RES[tricode]
# patches/ adds the terminal PO3 torsion that Rosetta's own patch omits
pyrosetta.init('-mute all -extra_patch_fa '
	'patches/thr_phos3.txt patches/tyr_phos4.txt', silent=True)

pose = pyrosetta.pose_from_sequence('AAA', 'fa_standard')
rts = ChemicalManager.get_instance().residue_type_set('fa_standard')
rt = rts.name_map(base)
if variant:
	rt = rts.get_residue_type_with_variant_added(
		rt, getattr(VariantType, variant))
pose.replace_residue(2, ResidueFactory.create_residue(rt), True)
n_chi = rt.nchi()

sfxn = get_score_function()
mm = MoveMap(); mm.set_bb(False); mm.set_chi(False); mm.set_chi(2, True)
mini = MinMover(mm, sfxn, 'lbfgs_armijo_nonmonotone', 0.01, True)
grid = [a * STEP for a in range(int(360.0 / STEP))]
# Fold a torsion onto its symmetry-unique range, treating an ordinary
# torsion as the 360 degree case. The wrap is essential: pose.chi() returns
# the accumulated angle, and the minimiser can wind a torsion past a full
# turn, so raw values may fall far outside -180..180.
fold = lambda a, p: ((a + (p or 360.0) / 2.0) % (p or 360.0)) - (p or 360.0) / 2.0
near = lambda a: min(range(3), key=lambda i:
	abs(((a - WELLS[i] + 180.0) % 360.0) - 180.0))

table, offsets = [], [0]
for i in range(36):
	for j in range(36):
		pose.set_phi(2, -180 + i * 10)
		pose.set_psi(2, -180 + j * 10)
		minima = []
		for combo in itertools.product(grid, repeat=n_chi):
			for k, c in enumerate(combo): pose.set_chi(k + 1, 2, c)
			mini.apply(pose)
			minima.append((tuple(fold(pose.chi(k + 1, 2), sym[k])
				for k in range(n_chi)), sfxn(pose)))
		wells = {}
		for chi, e in minima:
			key = tuple(1 if sym[k] else near(c) + 1
				for k, c in enumerate(chi))
			wells.setdefault(key, []).append((chi, e))
		rows = []
		for members in wells.values():
			members.sort(key=lambda m: m[1])
			best = members[0]
			sig = []
			for k in range(n_chi):
				v = [((m[0][k] - best[0][k] + 180.0) % 360.0) - 180.0
					for m in members]
				mu = sum(v) / len(v)
				sig.append(max(math.sqrt(
					sum((x - mu) ** 2 for x in v) / len(v)), 0.5))
			rows.append([len(members), best[1], list(best[0]), sig])
		lo = min(r[1] for r in rows)
		z = sum(math.exp(-(r[1] - lo) / KT) for r in rows)
		for r in rows: r[1] = math.exp(-(r[1] - lo) / KT) / z
		rows.sort(key=lambda r: -r[1])
		for c, p, chi, sig in rows:
			table.append([c, round(p, 6)] + [round(x, 2) for x in chi]
				+ [round(x, 2) for x in sig])
		offsets.append(len(table))

json.dump({'tricode': tricode, 'n_chi': n_chi, 'rotamers': {
	'columns': ['count', 'prob']
		+ ['chi%d' % (k + 1) for k in range(n_chi)]
		+ ['sig%d' % (k + 1) for k in range(n_chi)],
	'table': table, 'bin_offsets': offsets}}, sys.stdout)

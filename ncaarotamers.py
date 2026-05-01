#!/usr/bin/env python3
# ncaarotamers: backbone-dependent rotamer libraries for non-canonical
# amino acids. Three pipelines selectable via CLI flag:
#   --denovo  Tier 3, NN-pot constrained chi scan, laptop, minutes-hours
#   --md      Tier 2, NN-pot + explicit-water MD, 1-4 GPUs, 1-3 days
#   --dft     Tier 1, RESP + DFT + MD, HPC, 1-3 weeks
# Methodology and references: README.md.
# License: MIT.

import os
# Silence TorchANI's cuaev-extension warning. Must run BEFORE the
# torchani / openmmml imports below, since both pull in torchani at
# import time and the warning fires on first import.
os.environ.setdefault('TORCHANI_NO_WARN_EXTENSIONS', '1')

import argparse
import itertools
import json
import logging
import math
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import gemmi
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
from ase.optimize import LBFGS
from ase.calculators.calculator import Calculator, all_changes
from torchani.models import ANI2x

warnings.filterwarnings('ignore', category=UserWarning,
	module=r'torchani.*')

# Tier 2 (--md) requires OpenMM + openmm-ml. These are mid-weight deps
# that install reliably; failure here aborts the whole import because
# Tiers 2 and 3 share the NN-potential calculator infrastructure.
try:
	from openmm import (
		unit as mmunit,
		LangevinMiddleIntegrator,
		Platform as MMPlatform,
		CustomTorsionForce,
	)
	from openmm.app import (
		Topology as OMMTopology,
		Element as OMMElement,
		Modeller,
		Simulation,
		ForceField as MMForceField,
		PME, HBonds,
	)
	from openmmml import MLPotential
	HAVE_MD = True
except ImportError as _md_err:
	HAVE_MD = False
	_MD_ERR = _md_err

# Tier 1 (--dft) requires the full DFT stack: Psi4, OpenFF, mdtraj.
# These are heavy and platform-sensitive (psi4 wheels are flaky on some
# OS / arch combinations). Wrap in try/except so --denovo and --md still
# work on a laptop without these packages installed.
try:
	import psi4
	from openff.toolkit.topology import Molecule as OFFMolecule
	from openff.toolkit.typing.engines.smirnoff import (
		ForceField as OFFForceField)
	from openff.units import unit as offunit
	from openff.recharge.charges.resp import generate_resp_charges
	from openff.recharge.esp.psi4 import Psi4ESPSettings
	from openff.recharge.grids import LatticeGridSettings
	import openmm
	from openmm import app as mmapp
	from scipy.cluster.hierarchy import linkage, fcluster
	HAVE_DFT = True
except ImportError as _dft_err:
	HAVE_DFT = False
	_DFT_ERR = _dft_err

# rot_v1 schema constants -- match Pose database.json["Rotamer Library"].
PHI_START, PHI_STEP, PHI_N = -180, 10, 36
PSI_START, PSI_STEP, PSI_N = -180, 10, 36

# Boltzmann at 300 K.
T_K          = 300.0
KB_KCAL      = 0.001987
KT_KCAL      = KB_KCAL * T_K
KT_HARTREE   = T_K * 3.166811563e-6
HARTREE2KCAL = 627.5094740631
EV2KCAL      = 23.060541945329
KB_KJ_PER_MOL = 0.0083144626
KT_KJ_PER_MOL = KB_KJ_PER_MOL * T_K
KCAL2HARTREE = 1.0 / HARTREE2KCAL

# Chi-vector clustering and well-keep thresholds (shared).
CHI_CLUSTER_DEG = 30.0
WELL_MIN_PROB   = 0.03
SIGMA_FLOOR_DEG = 0.5
EFFECTIVE_N_FLOOR = 3.0
CANONICAL_WELLS_DEG = (-60.0, 60.0, 180.0)

# NN-potential constrained-scan parameters (Tiers 2, 3).
PHIPSI_K_EV_PER_RAD2 = 25.0
LBFGS_FMAX_EV_A = 0.05
LBFGS_MAX_STEPS = 250
HESS_DELTA_A    = 0.01
NN_MODEL = 'ani2x'

# DFT levels (Tier 1).
DFT_FUNCTIONAL = 'wb97x-d'
DFT_OPT_BASIS  = 'ma-def2-TZVP'
DFT_E_BASIS    = 'aug-cc-pVTZ'
RESP_BASIS     = '6-31G(d)'
PCM_SOLVENT    = 'water'
PCM_DIELECTRIC = 78.355
OPENFF_OFFXML  = 'openff-2.1.0.offxml'
PHI_PSI_RESTRAINT_K_OPT_KCAL = 1000.0
PHI_PSI_RESTRAINT_K_MD_KCAL  = 50.0

# MD parameters: Tier 1 (TIP4P-Ew, full grid) vs Tier 2 (TIP3P, top
# wells per bin). Tier 1 reproduces the original DFT pipeline's MD
# settings; Tier 2 trims them for the GPU-workstation budget.
DFT_WATER_MODEL_XML = 'tip4pew.xml'
DFT_WATER_PADDING_A = 12.0
DFT_ION_CONC_M      = 0.15
DFT_MD_TIMESTEP_FS  = 2.0
DFT_MD_HMR_STEP_FS  = 4.0
DFT_MD_EQUIL_NS     = 1.0
DFT_MD_FRICTION_PS  = 1.0
DFT_MD_TEMP_K       = 300.0
DFT_MD_PRESSURE_BAR = 1.0
DFT_MD_NS_PER_NODE  = 100.0
DFT_MD_REPLICATES   = 3
MD_TIMESTEP_FS    = 1.0
MD_FRICTION_PS    = 1.0
MD_FRAME_SAVE_PS  = 1.0
MD_EQUIL_PS       = 5.0
MD_PHIPSI_K_KJ    = 2400.0
MD_NS_PER_BIN     = 0.1
MD_REPLICATES     = 1
MD_TOP_WELLS      = 1
MD_PLATFORM       = 'CUDA'
MD_WATER_PADDING_NM = 0.8
MD_ION_CONC_M       = 0.15

# CIF bond-order codes.
CIF_BOND_TYPES = {
	'SING': 1, 'DOUB': 2, 'TRIP': 3, 'AROM': 4,
	'sing': 1, 'doub': 2, 'trip': 3, 'arom': 4,
	1: 1, 2: 2, 3: 3, 4: 4,
}

# Worker count for the chi scan ProcessPoolExecutor.
WORKERS_DENOVO = max(1, os.cpu_count() // 2)
WORKERS_DFT    = int(os.environ.get('NCAA_DFT_WORKERS', 64))

# Hard-coded canonical-residue chi chains for Tier 1 fast-path; NCAAs
# fall back to the generic walker.
_CHI_CHAINS_BY_RESIDUE = {
	'ARG': [('N','CA','CB','CG'), ('CA','CB','CG','CD'),
		('CB','CG','CD','NE'), ('CG','CD','NE','CZ')],
	'ASN': [('N','CA','CB','CG'), ('CA','CB','CG','OD1')],
	'ASP': [('N','CA','CB','CG'), ('CA','CB','CG','OD1')],
	'CYS': [('N','CA','CB','SG')],
	'GLN': [('N','CA','CB','CG'), ('CA','CB','CG','CD'),
		('CB','CG','CD','OE1')],
	'GLU': [('N','CA','CB','CG'), ('CA','CB','CG','CD'),
		('CB','CG','CD','OE1')],
	'HIS': [('N','CA','CB','CG'), ('CA','CB','CG','ND1')],
	'ILE': [('N','CA','CB','CG1'), ('CA','CB','CG1','CD1')],
	'LEU': [('N','CA','CB','CG'), ('CA','CB','CG','CD1')],
	'LYS': [('N','CA','CB','CG'), ('CA','CB','CG','CD'),
		('CB','CG','CD','CE'), ('CG','CD','CE','NZ')],
	'MET': [('N','CA','CB','CG'), ('CA','CB','CG','SD'),
		('CB','CG','SD','CE')],
	'PHE': [('N','CA','CB','CG'), ('CA','CB','CG','CD1')],
	'PRO': [('N','CA','CB','CG'), ('CA','CB','CG','CD'),
		('CB','CG','CD','N')],
	'SER': [('N','CA','CB','OG')],
	'THR': [('N','CA','CB','OG1')],
	'TRP': [('N','CA','CB','CG'), ('CA','CB','CG','CD1')],
	'TYR': [('N','CA','CB','CG'), ('CA','CB','CG','CD1')],
	'VAL': [('N','CA','CB','CG1')],
}

# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------

def setup_logging(name='ncaarotamers'):
	fmt = '%(asctime)s [%(levelname)s] %(message)s'
	log = logging.getLogger(name)
	log.setLevel(logging.INFO)
	log.handlers.clear()
	sh = logging.StreamHandler(sys.stderr)
	sh.setFormatter(logging.Formatter(fmt))
	log.addHandler(sh)
	return log


def _bin_index(angle, start, step, n):
	a = ((float(angle) - start) % (step * n)) / step
	return int(a + 0.5) % n


def _phi_psi_grid():
	nodes = []
	for i in range(PHI_N):
		for j in range(PSI_N):
			phi = PHI_START + i * PHI_STEP
			psi = PSI_START + j * PSI_STEP
			nodes.append((i, j, float(phi), float(psi)))
	return nodes


def _round(x, ndigits):
	r = round(float(x), ndigits)
	return 0.0 if r == 0.0 else r


def _wrap_deg(x):
	return ((x + 180.0) % 360.0) - 180.0


def _dihedral_deg(P, *idx):
	# Accepts either (P, p1, p2, p3, p4) flat index args or four 3-vectors.
	# Always returns the dihedral in degrees.
	if len(idx) == 4 and all(isinstance(i, (int, np.integer)) for i in idx):
		i, j, k, l = idx
		p1, p2, p3, p4 = P[i], P[j], P[k], P[l]
	else:
		p1, p2, p3, p4 = idx
	b1 = np.array(p2) - np.array(p1)
	b2 = np.array(p3) - np.array(p2)
	b3 = np.array(p4) - np.array(p3)
	n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
	nb2 = b2 / (np.linalg.norm(b2) + 1e-12)
	x = float(np.dot(n1, n2))
	y = float(np.dot(np.cross(n1, nb2), n2))
	return math.degrees(math.atan2(y, x))


def _chi_dist_linf(a, b):
	d = 0.0
	for x, y in zip(a, b):
		delta = abs(((x - y + 180.0) % 360.0) - 180.0)
		if delta > d:
			d = delta
	return d


def _state_index(chi):
	# Map chi (deg) to canonical well index: 1=g+, 2=t, 3=g-.
	x = ((chi + 180.0) % 360.0) - 180.0
	if -90 <= x < -30:
		return 3
	if 30 <= x < 90:
		return 1
	if -30 <= x < 30:
		return 2
	return 2


def _classify_well(angle_deg):
	# Same intent as _state_index but matches Tier 1's Dunbrack r-tuple
	# convention via nearest-canonical-well calculation.
	x = ((float(angle_deg) + 180.0) % 360.0) - 180.0
	dists = [
		(abs(((x - 60.0 + 180.0) % 360.0) - 180.0), 1),
		(abs(((x - 180.0 + 180.0) % 360.0) - 180.0), 2),
		(abs(((x + 60.0 + 180.0) % 360.0) - 180.0), 3),
	]
	dists.sort()
	return dists[0][1]


def parse_cif_and_build_tripeptide(cif_path, tricode, log):
	# Parse CIF, build RDKit mol, attach ACE / NME caps with chemically
	# sensible cap geometry, AddHs, UFF-relax cap atoms only.
	# Returns (capped_mol, label_to_idx, ace_C_idx, nme_N_idx).
	log.info(f'Parsing CIF: {cif_path}')
	doc = gemmi.cif.read(str(cif_path))
	block = list(doc)[0]
	atoms_loop = block.find('_chem_comp_atom.', [
		'atom_id', 'type_symbol', 'charge',
		'pdbx_model_Cartn_x_ideal',
		'pdbx_model_Cartn_y_ideal',
		'pdbx_model_Cartn_z_ideal',
		'pdbx_leaving_atom_flag',
	])
	bonds_loop = block.find('_chem_comp_bond.', [
		'atom_id_1', 'atom_id_2', 'value_order',
	])
	rw = Chem.RWMol()
	label_to_idx = {}
	coords = []
	leaving = set()
	for row in atoms_loop:
		lab = row.str(0).strip()
		elt = row.str(1).strip()
		try:
			charge = int(float(row.str(2) or '0'))
		except (ValueError, TypeError):
			charge = 0
		x = float(row.str(3)); y = float(row.str(4)); z = float(row.str(5))
		leav = row.str(6).strip().upper() == 'Y'
		if elt == 'D':
			elt = 'H'
		a = Chem.Atom(elt)
		a.SetFormalCharge(charge)
		idx = rw.AddAtom(a)
		rw.GetAtomWithIdx(idx).SetProp('cif_label', lab)
		label_to_idx[lab] = idx
		coords.append((x, y, z))
		if leav:
			leaving.add(lab)
	bo_map = {
		1: Chem.BondType.SINGLE,
		2: Chem.BondType.DOUBLE,
		3: Chem.BondType.TRIPLE,
		4: Chem.BondType.AROMATIC,
	}
	for row in bonds_loop:
		a = row.str(0).strip(); b = row.str(1).strip()
		order = CIF_BOND_TYPES.get(row.str(2).strip(), 1)
		if a in label_to_idx and b in label_to_idx:
			rw.AddBond(label_to_idx[a], label_to_idx[b], bo_map[order])
	conf = Chem.Conformer(rw.GetNumAtoms())
	for k, xyz in enumerate(coords):
		conf.SetAtomPosition(k, Chem.rdGeometry.Point3D(*xyz))
	rw.AddConformer(conf)
	mol = rw.GetMol()
	Chem.SanitizeMol(mol)
	log.info(f'  parsed {mol.GetNumAtoms()} atoms, '
		f'{mol.GetNumBonds()} bonds (leaving: {sorted(leaving)})')
	for req in ('N', 'CA', 'C', 'O'):
		if req not in label_to_idx:
			raise ValueError(
				f'CIF for {tricode} missing backbone atom {req!r}')
	# Drop leaving atoms (PDB chem-comp convention: removed at peptide
	# bond formation -- N-terminal H, C-terminal OXT/HXT).
	rw = Chem.RWMol(mol)
	drop_ids = sorted([label_to_idx[l] for l in leaving], reverse=True)
	for aid in drop_ids:
		rw.RemoveAtom(aid)
	label_to_idx = {}
	for k, atom in enumerate(rw.GetAtoms()):
		lab = atom.GetPropsAsDict().get('cif_label', None)
		if lab is not None:
			label_to_idx[lab] = k
	i_N = label_to_idx['N']; i_CA = label_to_idx['CA']
	i_C = label_to_idx['C']; i_O = label_to_idx['O']
	# ACE = -C(=O)CH3 attaches to N; NME = -NHCH3 attaches to C.
	ace_C  = rw.AddAtom(Chem.Atom('C'))
	ace_O  = rw.AddAtom(Chem.Atom('O'))
	ace_Me = rw.AddAtom(Chem.Atom('C'))
	rw.AddBond(ace_C, ace_O,  Chem.BondType.DOUBLE)
	rw.AddBond(ace_C, ace_Me, Chem.BondType.SINGLE)
	rw.AddBond(ace_C, i_N,    Chem.BondType.SINGLE)
	nme_N  = rw.AddAtom(Chem.Atom('N'))
	nme_Me = rw.AddAtom(Chem.Atom('C'))
	rw.AddBond(nme_N, nme_Me, Chem.BondType.SINGLE)
	rw.AddBond(nme_N, i_C,    Chem.BondType.SINGLE)
	# Place cap atoms at chemically sensible positions BEFORE AddHs.
	conf = rw.GetConformer()
	p_N = np.array([conf.GetAtomPosition(i_N).x,
		conf.GetAtomPosition(i_N).y, conf.GetAtomPosition(i_N).z])
	p_CA = np.array([conf.GetAtomPosition(i_CA).x,
		conf.GetAtomPosition(i_CA).y, conf.GetAtomPosition(i_CA).z])
	p_C = np.array([conf.GetAtomPosition(i_C).x,
		conf.GetAtomPosition(i_C).y, conf.GetAtomPosition(i_C).z])
	v_NCA = p_N - p_CA
	v_NCA = v_NCA / (np.linalg.norm(v_NCA) + 1e-9)
	p_ace_C  = p_N + 1.33 * v_NCA
	p_ace_O  = p_ace_C + np.array([0.0, 1.23, 0.0])
	p_ace_Me = p_ace_C + 1.51 * v_NCA
	v_CCA = p_C - p_CA
	v_CCA = v_CCA / (np.linalg.norm(v_CCA) + 1e-9)
	p_nme_N  = p_C + 1.33 * v_CCA
	p_nme_Me = p_nme_N + 1.45 * v_CCA
	for idx, p in [(ace_C, p_ace_C), (ace_O, p_ace_O),
			(ace_Me, p_ace_Me), (nme_N, p_nme_N),
			(nme_Me, p_nme_Me)]:
		conf.SetAtomPosition(idx, Chem.rdGeometry.Point3D(
			float(p[0]), float(p[1]), float(p[2])))
	capped_no_h = rw.GetMol()
	Chem.SanitizeMol(capped_no_h)
	capped = Chem.AddHs(capped_no_h, addCoords=True)
	# UFF-relax cap atoms only; tolerate failure on charged residues
	# where UFF's BFGS can diverge.
	try:
		ff = AllChem.UFFGetMoleculeForceField(capped)
		if ff is not None:
			frozen = []
			for k, atom in enumerate(capped.GetAtoms()):
				lab = atom.GetPropsAsDict().get('cif_label', None)
				if lab is not None and atom.GetSymbol() != 'H':
					frozen.append(k)
			for k in frozen:
				ff.AddFixedPoint(k)
			ff.Minimize(maxIts=2000)
	except Exception:
		pass
	label_to_idx = {}
	for k, atom in enumerate(capped.GetAtoms()):
		lab = atom.GetPropsAsDict().get('cif_label', None)
		if lab is not None:
			label_to_idx[lab] = k
	return capped, label_to_idx, ace_C, nme_N


def rdkit_to_ase(rd_mol):
	conf = rd_mol.GetConformer()
	syms, pos = [], []
	for a in rd_mol.GetAtoms():
		syms.append(a.GetSymbol())
		p = conf.GetAtomPosition(a.GetIdx())
		pos.append([p.x, p.y, p.z])
	return Atoms(symbols=syms, positions=np.asarray(pos))


def make_nn_calculator(model=NN_MODEL):
	# ANI-2x covers H, C, N, O, F, S, Cl. Other models can be added by
	# extending this dispatcher; for the v1 release ANI-2x is the only
	# verified backend.
	if model.lower() in ('ani2x', 'ani-2x'):
		return ANI2x().ase()
	raise ValueError(f'Unknown NN potential: {model!r}')


def auto_detect_chi_axes(rd_mol, label_to_idx, log):
	# BFS the side-chain heavy-atom graph from CB outward in CIF-ordinal
	# order; emit consecutive 4-atom dihedrals.
	from collections import defaultdict
	for n in ('N', 'CA', 'CB', 'C', 'O'):
		if n not in label_to_idx:
			raise KeyError(
				f'CIF missing required atom {n!r}; auto-detect supports '
				f'standard amino acid backbones only.')
	cif_ord = {}
	for k, atom in enumerate(rd_mol.GetAtoms()):
		lab = atom.GetPropsAsDict().get('cif_label', None)
		if lab is not None:
			cif_ord[lab] = k
	adj = defaultdict(list)
	for bond in rd_mol.GetBonds():
		a, b = bond.GetBeginAtom(), bond.GetEndAtom()
		if a.GetSymbol() == 'H' or b.GetSymbol() == 'H':
			continue
		la = a.GetPropsAsDict().get('cif_label', None)
		lb = b.GetPropsAsDict().get('cif_label', None)
		if la is None or lb is None:
			continue
		adj[la].append(lb)
		adj[lb].append(la)
	excluded = {'N', 'CA', 'C', 'O', 'OXT'}
	mc, visited = [], set(excluded) | {'CA'}
	cur = 'CB'
	while cur is not None:
		mc.append(cur)
		visited.add(cur)
		nbrs = [n for n in adj.get(cur, [])
			if n not in visited and n in cif_ord]
		cur = min(nbrs, key=lambda n: cif_ord[n]) if nbrs else None
	full_chain = ['N', 'CA'] + mc
	chis = [tuple(full_chain[i:i+4])
		for i in range(len(full_chain) - 3)]
	log.info(f'Auto-detected {len(chis)} chi axes:')
	for k, c in enumerate(chis):
		log.info(f'  chi{k+1}: {c}')
	return [list(c) for c in chis]


def resolve_chi_axes(label_to_idx, chi_atom_names):
	out = []
	for axis in chi_atom_names:
		idxs = []
		for an in axis:
			if an not in label_to_idx:
				raise KeyError(
					f'chi-axis atom {an!r} not in CIF labels: '
					f'{sorted(label_to_idx)}')
			idxs.append(label_to_idx[an])
		out.append(tuple(idxs))
	return out


def resolve_phi_psi_atoms(label_to_idx, ace_C_idx, nme_N_idx):
	return ((ace_C_idx, label_to_idx['N'], label_to_idx['CA'],
			label_to_idx['C']),
		(label_to_idx['N'], label_to_idx['CA'],
			label_to_idx['C'], nme_N_idx))


def make_restrained_calculator_multi_k(base_calc, restraints):
	# Wrap base_calc with harmonic dihedral restraints; each carries its
	# own spring constant. Forces from the restraint are computed via
	# central-difference of the restraint energy.
	base = base_calc

	class _Wrapped(Calculator):
		implemented_properties = ['energy', 'forces']
		def calculate(self, atoms=None, properties=['energy'],
				system_changes=all_changes):
			Calculator.calculate(self, atoms, properties, system_changes)
			base.calculate(atoms, properties, system_changes)
			E = float(base.results['energy'])
			F = np.array(base.results['forces'],
				dtype=np.float64).copy()
			eps = 1e-4
			P = atoms.positions
			for (i, j, kk, l, target_deg, k_ev) in restraints:
				phi = _dihedral_deg(P, i, j, kk, l)
				d = ((phi - target_deg + 180.0) % 360.0) - 180.0
				d_rad = math.radians(d)
				E += 0.5 * k_ev * d_rad * d_rad
				for atom_idx in (i, j, kk, l):
					for axis in range(3):
						p_save = P[atom_idx, axis]
						P[atom_idx, axis] = p_save + eps
						phi_p = _dihedral_deg(P, i, j, kk, l)
						d_p = ((phi_p - target_deg + 180.0)
							% 360.0) - 180.0
						E_p = 0.5 * k_ev * (math.radians(d_p) ** 2)
						P[atom_idx, axis] = p_save - eps
						phi_m = _dihedral_deg(P, i, j, kk, l)
						d_m = ((phi_m - target_deg + 180.0)
							% 360.0) - 180.0
						E_m = 0.5 * k_ev * (math.radians(d_m) ** 2)
						P[atom_idx, axis] = p_save
						F[atom_idx, axis] -= (E_p - E_m) / (2.0 * eps)
			self.results['energy'] = E
			self.results['forces'] = F
	return _Wrapped()


def emit_rot_v1(grid, n_chi, tricode, chi_axes_names, model_name,
		method_extra, out_path, log):
	# Common rot_v1 emitter used by all three pipelines. Each pipeline
	# writes its own 'method' block via method_extra.
	cols = (
		[f'r{k+1}' for k in range(n_chi)]
		+ ['count', 'prob']
		+ [f'chi{k+1}' for k in range(n_chi)]
		+ [f'sig{k+1}' for k in range(n_chi)])
	bins = [[] for _ in range(PHI_N * PSI_N)]
	top_chi = [[None] * PSI_N for _ in range(PHI_N)]
	for (i, j), rec in grid.items():
		wells = rec.get('wells') or []
		wells_sorted = sorted(wells, key=lambda w: -w['prob'])
		for w in wells_sorted:
			row = []
			for k in range(n_chi):
				row.append(int(_state_index(float(w['chi'][k]))))
			row.append(int(round(w['prob'] * 1e6)))
			row.append(round(float(w['prob']), 6))
			for k in range(n_chi):
				row.append(round(float(w['chi'][k]), 4))
			for k in range(n_chi):
				row.append(round(max(float(w['sigma'][k]),
					SIGMA_FLOOR_DEG), 4))
			bins[i * PSI_N + j].append(row)
		top_chi[i][j] = ([round(float(c), 4)
			for c in wells_sorted[0]['chi']]
			if wells_sorted else [0.0] * n_chi)
	table = []
	bin_offsets = [0] * (PHI_N * PSI_N + 1)
	for k, b in enumerate(bins):
		bin_offsets[k] = len(table)
		table.extend(b)
	bin_offsets[PHI_N * PSI_N] = len(table)
	method = {
		'pipeline': 'ncaarotamers.py',
		'model':    model_name,
		'phi_grid': [PHI_START, PHI_STEP, PHI_N],
		'psi_grid': [PSI_START, PSI_STEP, PSI_N],
		'chi_cluster_deg': CHI_CLUSTER_DEG,
		'well_min_prob':   WELL_MIN_PROB,
		'sigma_floor':     SIGMA_FLOOR_DEG,
		'temperature_K':   T_K,
		'chi_axes':        [list(a) for a in chi_axes_names],
	}
	if method_extra:
		method.update(method_extra)
	entry = {
		'tricode':  tricode,
		'n_chi':    n_chi,
		'rotamers': {
			'columns':     cols,
			'table':       table,
			'bin_offsets': bin_offsets,
			'top_chi':     top_chi,
		},
		'densities': None,
		'method':    method,
		'metadata': {
			'phi_grid':  [PHI_START, PHI_STEP, PHI_N],
			'psi_grid':  [PSI_START, PSI_STEP, PSI_N],
			'sigma_floor_deg': SIGMA_FLOOR_DEG,
		},
	}
	with open(out_path, 'w') as fh:
		json.dump(entry, fh, separators=(',', ':'))
	log.info(f'wrote {out_path} '
		f'({os.path.getsize(out_path)/1e6:.2f} MB, '
		f'{len(table)} table rows)')
	return entry


# ----------------------------------------------------------------------
# Tier 3 (--denovo) -- NN-pot constrained chi scan, gas-phase, no MD.
# ----------------------------------------------------------------------

def _ang_cluster_means(chi_samples, weights, n_components, random_state):
	from sklearn.mixture import BayesianGaussianMixture
	if chi_samples.ndim == 1:
		chi_samples = chi_samples[:, None]
	N, D = chi_samples.shape
	X = np.empty((N, 2 * D), dtype=np.float64)
	for j in range(D):
		X[:, 2*j] = np.cos(np.deg2rad(chi_samples[:, j]))
		X[:, 2*j+1] = np.sin(np.deg2rad(chi_samples[:, j]))
	Neff_target = max(200, min(int(weights.sum() * 50), 5000))
	p = weights / weights.sum()
	rng = np.random.default_rng(random_state)
	idx = rng.choice(N, size=Neff_target, p=p, replace=True)
	bgmm = BayesianGaussianMixture(
		n_components=n_components,
		covariance_type='full',
		weight_concentration_prior_type='dirichlet_process',
		weight_concentration_prior=1e-2,
		max_iter=400, reg_covar=1e-4,
		random_state=random_state, init_params='kmeans')
	with np.errstate(divide='ignore', invalid='ignore'):
		bgmm.fit(X[idx])
	means, sigmas, pops = [], [], []
	for k in range(n_components):
		w = bgmm.weights_[k]
		if w < WELL_MIN_PROB:
			continue
		mu = bgmm.means_[k]
		cov = bgmm.covariances_[k]
		chi_mu, chi_sig = [], []
		for j in range(D):
			cx, sy = mu[2*j], mu[2*j+1]
			chi_mu.append(math.degrees(math.atan2(sy, cx)))
			cb = cov[2*j:2*j+2, 2*j:2*j+2]
			t = np.array([-sy, cx])
			t = t / (np.linalg.norm(t) + 1e-12)
			var_tan = float(t @ cb @ t)
			s = min(math.sqrt(max(var_tan, 0.0)), 0.999)
			sigma_rad = math.asin(s) if s < 0.999 else math.pi / 2
			chi_sig.append(max(math.degrees(sigma_rad),
				SIGMA_FLOOR_DEG))
		means.append(chi_mu); sigmas.append(chi_sig); pops.append(float(w))
	s = sum(pops) or 1.0
	pops = [p / s for p in pops]
	order = np.argsort(-np.array(pops))
	return ([means[i] for i in order],
		[sigmas[i] for i in order],
		[pops[i] for i in order])


def _scan_one_bin(args):
	# Two-pass restrained scan at one (phi, psi) bin: stiff chi seed in
	# pass 1, weak chi restraint in pass 2 to preserve rotamer-well
	# topology while letting chi relax to the local minimum.
	(atoms_template_dict, model_name, phi_deg, psi_deg,
		phi_atoms, psi_atoms, chi_atom_indices, n_shared_chi,
		parent_chi_at_bin, novel_chi_seeds, k_phipsi_ev) = args
	base_calc = make_nn_calculator(model_name)
	n_chi = len(chi_atom_indices)
	n_novel = n_chi - n_shared_chi
	shared_axes = chi_atom_indices[:n_shared_chi]
	novel_axes = chi_atom_indices[n_shared_chi:]
	k_chi_shared_ev = k_phipsi_ev * 8.0
	k_chi_novel_pass1_ev = k_phipsi_ev * 4.0
	k_chi_novel_pass2_ev = k_chi_novel_pass1_ev / 40.0
	if n_novel == 0:
		return {'phi': float(phi_deg), 'psi': float(psi_deg),
			'wells': [{'chi': list(parent_chi_at_bin),
				'sigma': [SIGMA_FLOOR_DEG] * n_chi,
				'energy': 0.0, 'prob': 1.0}]}
	seed_iter = novel_chi_seeds if novel_chi_seeds else [
		[180.0] * n_novel]
	wells_raw = []
	for novel_seed in seed_iter:
		atoms = Atoms(
			symbols=atoms_template_dict['symbols'],
			positions=np.array(atoms_template_dict['positions']))
		restraints_phipsi = [
			(phi_atoms[0], phi_atoms[1], phi_atoms[2], phi_atoms[3],
				float(phi_deg), k_phipsi_ev),
			(psi_atoms[0], psi_atoms[1], psi_atoms[2], psi_atoms[3],
				float(psi_deg), k_phipsi_ev),
		]
		restraints_shared = [
			(ax[0], ax[1], ax[2], ax[3],
				float(parent_chi_at_bin[k]), k_chi_shared_ev)
			for k, ax in enumerate(shared_axes)
		]
		restraints_novel_p1 = [
			(ax[0], ax[1], ax[2], ax[3],
				float(novel_seed[k]), k_chi_novel_pass1_ev)
			for k, ax in enumerate(novel_axes)
		]
		atoms.calc = make_restrained_calculator_multi_k(base_calc,
			restraints_phipsi + restraints_shared + restraints_novel_p1)
		try:
			LBFGS(atoms, logfile=None).run(
				fmax=LBFGS_FMAX_EV_A * 2,
				steps=LBFGS_MAX_STEPS // 2)
		except Exception:
			continue
		restraints_novel_p2 = [
			(ax[0], ax[1], ax[2], ax[3],
				float(novel_seed[k]), k_chi_novel_pass2_ev)
			for k, ax in enumerate(novel_axes)
		]
		atoms.calc = make_restrained_calculator_multi_k(base_calc,
			restraints_phipsi + restraints_shared + restraints_novel_p2)
		try:
			LBFGS(atoms, logfile=None).run(
				fmax=LBFGS_FMAX_EV_A,
				steps=LBFGS_MAX_STEPS)
		except Exception:
			continue
		P = atoms.positions
		chi_final = [_dihedral_deg(P, *ax) for ax in chi_atom_indices]
		atoms.calc = base_calc
		try:
			E_eV = float(atoms.get_potential_energy())
		except Exception:
			continue
		wells_raw.append({
			'chi':       chi_final,
			'energy':    E_eV * EV2KCAL,
			'positions': atoms.positions.copy(),
		})
	if not wells_raw:
		return {'phi': float(phi_deg), 'psi': float(psi_deg),
			'wells': [], 'error': 'no_converged'}
	# Cluster minima by chi-vector L_inf distance.
	chi_arr = np.array([w['chi'] for w in wells_raw])
	n = len(chi_arr)
	labels = list(range(n))
	for i in range(n):
		for j in range(i):
			if _chi_dist_linf(chi_arr[i], chi_arr[j]) < CHI_CLUSTER_DEG:
				labels[i] = labels[j]
				break
	clusters = {}
	for w, lab in zip(wells_raw, labels):
		clusters.setdefault(lab, []).append(w)
	merged = []
	for members in clusters.values():
		members.sort(key=lambda r: r['energy'])
		rep = members[0]
		spreads = []
		for ax_idx in range(len(rep['chi'])):
			vals = np.array([m['chi'][ax_idx] for m in members])
			if len(vals) > 1:
				centered = (((vals - rep['chi'][ax_idx] + 180.0)
					% 360.0) - 180.0)
				spreads.append(max(float(np.std(centered)),
					SIGMA_FLOOR_DEG))
			else:
				spreads.append(SIGMA_FLOOR_DEG)
		merged.append({
			'chi':    [round(float(c), 4) for c in rep['chi']],
			'sigma':  [round(float(s), 4) for s in spreads],
			'energy': float(rep['energy']),
		})
	E = np.array([w['energy'] for w in merged])
	E -= E.min()
	Z = np.exp(-E / KT_KCAL).sum()
	for w, e in zip(merged, E):
		w['prob'] = float(np.exp(-e / KT_KCAL) / Z)
	merged = [w for w in merged if w['prob'] >= WELL_MIN_PROB]
	s = sum(w['prob'] for w in merged) or 1.0
	for w in merged:
		w['prob'] = w['prob'] / s
	merged.sort(key=lambda w: -w['prob'])
	return {'phi': float(phi_deg), 'psi': float(psi_deg),
		'wells': merged}


def novel_chi_seeds_canonical(n_novel):
	if n_novel == 0:
		return []
	if n_novel > 4:
		seeds = list(itertools.product(CANONICAL_WELLS_DEG, repeat=4))
		return [list(s) + [180.0] * (n_novel - 4) for s in seeds]
	return [list(s) for s in itertools.product(
		CANONICAL_WELLS_DEG, repeat=n_novel)]


def run_grid(atoms_template_dict, model_name, phi_atoms, psi_atoms,
		chi_atom_indices, n_chi, n_shared_chi,
		parent_top_chi_grid, n_workers, log):
	n_novel = n_chi - n_shared_chi
	novel_seeds = novel_chi_seeds_canonical(n_novel)
	nodes = []
	for i in range(PHI_N):
		for j in range(PSI_N):
			phi = PHI_START + i * PHI_STEP
			psi = PSI_START + j * PSI_STEP
			parent_chi_at_bin = []
			if n_shared_chi > 0 and parent_top_chi_grid is not None:
				p = list(parent_top_chi_grid[i][j])
				if len(p) < n_shared_chi:
					p = p + [180.0] * (n_shared_chi - len(p))
				else:
					p = p[:n_shared_chi]
				parent_chi_at_bin = p
			nodes.append((i, j, phi, psi, parent_chi_at_bin))
	log.info(f'Scanning {len(nodes)} (phi, psi) bins | n_chi={n_chi} '
		f'(shared={n_shared_chi}, novel={n_novel}) | '
		f'{len(novel_seeds)} novel-chi seeds/bin | '
		f'{n_workers} workers')
	args_list = []
	for (i, j, phi, psi, parent_chi_at_bin) in nodes:
		args_list.append((atoms_template_dict, model_name, phi, psi,
			phi_atoms, psi_atoms, chi_atom_indices,
			n_shared_chi, parent_chi_at_bin, novel_seeds,
			PHIPSI_K_EV_PER_RAD2))
	grid = {}
	t0 = time.time()
	n_done = 0
	with ProcessPoolExecutor(max_workers=n_workers) as ex:
		futs = {ex.submit(_scan_one_bin, a): k
			for k, a in enumerate(args_list)}
		for fut in as_completed(futs):
			k = futs[fut]
			i = k // PSI_N; j = k % PSI_N
			try:
				grid[(i, j)] = fut.result()
			except Exception as e:
				log.warning(f'  bin ({i},{j}) failed: {e}')
				grid[(i, j)] = {
					'phi': float(PHI_START + i * PHI_STEP),
					'psi': float(PSI_START + j * PSI_STEP),
					'wells': [], 'error': str(e)}
			n_done += 1
			if n_done % 50 == 0:
				el = time.time() - t0
				eta = el / n_done * (len(nodes) - n_done)
				log.info(f'  progress {n_done}/{len(nodes)} '
					f'({100*n_done/len(nodes):.1f}%); '
					f'elapsed {el/60:.1f} min; '
					f'ETA {eta/60:.1f} min')
	return grid


def fill_empty_bins_with_global(grid, atoms_template_dict, model_name,
		chi_atom_indices, n_chi, n_shared_chi, log):
	log.info('Backbone-independent fallback scan for empty bins')
	n_novel = n_chi - n_shared_chi
	novel_seeds = novel_chi_seeds_canonical(n_novel)
	parent_chi_global = [180.0] * n_shared_chi
	args = (atoms_template_dict, model_name, 0.0, 0.0,
		(0, 0, 0, 0), (0, 0, 0, 0), chi_atom_indices,
		n_shared_chi, parent_chi_global, novel_seeds, 0.0)
	res = _scan_one_bin(args)
	fb_wells = res['wells']
	log.info(f'  global wells: {len(fb_wells)}')
	n_fill = 0
	for (i, j), rec in grid.items():
		if not rec.get('wells'):
			rec['wells'] = [dict(w) for w in fb_wells]
			rec['source'] = 'global'
			n_fill += 1
		else:
			rec['source'] = rec.get('source', 'kernel')
	log.info(f'  filled {n_fill} empty bins from global fallback')


def pipeline_denovo(cif, tricode, out_path, log):
	t0 = time.time()
	log.info('=' * 60)
	log.info(f'Tier 3 (--denovo) pipeline: cif={cif}, tricode={tricode}, '
		f'model={NN_MODEL}')
	capped, label_to_idx, ace_C, nme_N = (
		parse_cif_and_build_tripeptide(cif, tricode, log))
	chi_axes_names = auto_detect_chi_axes(capped, label_to_idx, log)
	n_chi = len(chi_axes_names)
	chi_atom_indices = resolve_chi_axes(label_to_idx, chi_axes_names)
	phi_atoms, psi_atoms = resolve_phi_psi_atoms(
		label_to_idx, ace_C, nme_N)
	ase_atoms = rdkit_to_ase(capped)
	atoms_template_dict = {
		'symbols':   list(ase_atoms.get_chemical_symbols()),
		'positions': ase_atoms.positions.tolist(),
	}
	log.info(f'Tripeptide ASE: {len(ase_atoms)} atoms; '
		f'phi={phi_atoms}, psi={psi_atoms}')
	smoke = ase_atoms.copy()
	smoke.calc = make_nn_calculator(NN_MODEL)
	E0 = float(smoke.get_potential_energy())
	log.info(f'Smoke E0 = {E0:.4f} eV')
	grid = run_grid(atoms_template_dict, NN_MODEL,
		phi_atoms, psi_atoms, chi_atom_indices,
		n_chi, n_shared_chi=0, parent_top_chi_grid=None,
		n_workers=WORKERS_DENOVO, log=log)
	fill_empty_bins_with_global(
		grid, atoms_template_dict, NN_MODEL,
		chi_atom_indices, n_chi, n_shared_chi=0, log=log)
	method_extra = {
		'tier':       '3',
		'kind':       f'NN-potential ({NN_MODEL}) constrained '
			f'(phi, psi) chi scan, gas-phase',
		'cif':        os.path.basename(cif),
		'n_workers':  WORKERS_DENOVO,
		'phipsi_k_ev_per_rad2': PHIPSI_K_EV_PER_RAD2,
		'lbfgs_fmax_ev_a':       LBFGS_FMAX_EV_A,
		'lbfgs_max_steps':       LBFGS_MAX_STEPS,
		'solvent':   'gas-phase',
		'citations': [
			'Devereux et al., JCTC 2020 (ANI-2x)',
			'Larsen et al., JPCM 2017 (ASE)',
			'Shapovalov & Dunbrack, Structure 2011 (BBDEP)',
		],
	}
	emit_rot_v1(grid, n_chi, tricode, chi_axes_names, NN_MODEL,
		method_extra, out_path, log)
	log.info(f'TOTAL wall-time: {(time.time() - t0)/60:.1f} min')


# ----------------------------------------------------------------------
# Tier 2 (--md) -- NN-pot scan + Hessian sigmas + explicit-water MD
# refinement at top wells via openmm-ml MLPotential.
# ----------------------------------------------------------------------

def _wilson_b_dihedral_md(xyz, axis):
	i, j, k, l = axis
	N = len(xyz)
	out = np.zeros(3 * N, dtype=np.float64)
	eps = 1e-4
	P = np.array(xyz, dtype=np.float64)
	for atom_idx in (i, j, k, l):
		for d in range(3):
			save = P[atom_idx, d]
			P[atom_idx, d] = save + eps
			phi_p = _dihedral_deg(P, i, j, k, l)
			P[atom_idx, d] = save - eps
			phi_m = _dihedral_deg(P, i, j, k, l)
			P[atom_idx, d] = save
			d_phi = ((phi_p - phi_m + 180.0) % 360.0) - 180.0
			out[3 * atom_idx + d] = math.radians(d_phi) / (2 * eps)
	return out


def _numerical_hessian(atoms, calc, delta=HESS_DELTA_A):
	N = len(atoms)
	H = np.zeros((3 * N, 3 * N), dtype=np.float64)
	base_pos = atoms.positions.copy()
	for i in range(N):
		for d in range(3):
			atoms.positions = base_pos.copy()
			atoms.positions[i, d] += delta
			atoms.calc = calc
			F_plus = atoms.get_forces().copy()
			atoms.positions = base_pos.copy()
			atoms.positions[i, d] -= delta
			atoms.calc = calc
			F_minus = atoms.get_forces().copy()
			H[3*i + d, :] = -((F_plus - F_minus) / (2 * delta)).reshape(-1)
	atoms.positions = base_pos
	H = 0.5 * (H + H.T)
	return H


def chi_sigmas_from_hessian(atoms, base_calc, chi_axes,
		k_min_floor_kcal=0.5, log=None):
	if log:
		log.info('  computing numerical Hessian (NN-potential)')
	H_eV_per_A2 = _numerical_hessian(atoms, base_calc)
	xyz = atoms.positions
	sigmas = []
	for axis in chi_axes:
		B = _wilson_b_dihedral_md(xyz, axis)
		BtHB = float(B @ H_eV_per_A2 @ B)
		BtB = float(B @ B)
		if BtB <= 0 or BtHB <= 0:
			sigmas.append(min(60.0, max(SIGMA_FLOOR_DEG,
				k_min_floor_kcal)))
			continue
		k_eV = BtHB / BtB
		k_kcal = max(k_eV * EV2KCAL, k_min_floor_kcal)
		sigma_rad = math.sqrt(KT_KCAL / k_kcal)
		sigmas.append(min(60.0, max(SIGMA_FLOOR_DEG,
			math.degrees(sigma_rad))))
	return sigmas


def md_validate_one_well(rd_mol, ase_atoms_at_well, model_name,
		phi_atoms, psi_atoms, phi_deg, psi_deg, chi_atom_indices,
		md_ns, md_replicates, platform_name, log):
	# Solvate Ace-X-Nme around the well, run NN-pot MD with (phi, psi)
	# restrained, harvest chi samples.
	if not HAVE_MD:
		raise RuntimeError(
			f'--md requires openmm + openmmml: {_MD_ERR}')
	top = OMMTopology()
	chain = top.addChain()
	res = top.addResidue('LIG', chain)
	omm_atoms = []
	for atom in rd_mol.GetAtoms():
		elt = OMMElement.getBySymbol(atom.GetSymbol())
		omm_atoms.append(top.addAtom(atom.GetSymbol(), elt, res))
	for bond in rd_mol.GetBonds():
		i = bond.GetBeginAtomIdx()
		j = bond.GetEndAtomIdx()
		top.addBond(omm_atoms[i], omm_atoms[j])
	positions = mmunit.Quantity(
		ase_atoms_at_well.positions.copy(), mmunit.angstrom)
	modeller = Modeller(top, positions)
	water_ff = None
	for ff_name in ('tip3p.xml', 'amber19/tip3p.xml',
			'amber14/tip3p.xml'):
		try:
			water_ff = MMForceField(ff_name)
			break
		except Exception:
			continue
	if water_ff is None:
		log.warning('  no TIP3P FF found; gas-phase fallback')
	else:
		try:
			modeller.addSolvent(
				water_ff,
				padding=MD_WATER_PADDING_NM * mmunit.nanometer,
				ionicStrength=MD_ION_CONC_M * mmunit.molar,
				model='tip3p')
		except Exception as e:
			log.warning(f'  addSolvent failed ({e}); gas-phase fallback')
	n_solute = rd_mol.GetNumAtoms()
	solute_indices = list(range(n_solute))
	mp = MLPotential(model_name)
	if (water_ff is not None
			and modeller.topology.getNumAtoms() > n_solute):
		try:
			classical = water_ff.createSystem(
				modeller.topology,
				nonbondedMethod=PME,
				nonbondedCutoff=1.0 * mmunit.nanometer,
				constraints=HBonds)
			system = mp.createMixedSystem(
				modeller.topology, classical, solute_indices)
		except Exception as e:
			log.warning(f'  createMixedSystem failed ({e}); '
				f'gas-phase fallback')
			system = mp.createSystem(modeller.topology)
	else:
		system = mp.createSystem(modeller.topology)
	for (a, b, c, d, target_deg) in [
			(phi_atoms[0], phi_atoms[1], phi_atoms[2], phi_atoms[3],
				float(phi_deg)),
			(psi_atoms[0], psi_atoms[1], psi_atoms[2], psi_atoms[3],
				float(psi_deg))]:
		f = CustomTorsionForce('0.5*k*(theta - theta0)^2')
		f.addPerTorsionParameter('k')
		f.addPerTorsionParameter('theta0')
		f.addTorsion(a, b, c, d,
			[MD_PHIPSI_K_KJ, math.radians(target_deg)])
		system.addForce(f)
	integrator = LangevinMiddleIntegrator(
		T_K * mmunit.kelvin,
		MD_FRICTION_PS / mmunit.picosecond,
		MD_TIMESTEP_FS * mmunit.femtosecond)
	try:
		plat = MMPlatform.getPlatformByName(platform_name)
	except Exception:
		plat = None
	sim = (Simulation(modeller.topology, system, integrator, plat)
		if plat else
		Simulation(modeller.topology, system, integrator))
	sim.context.setPositions(modeller.positions)
	try:
		sim.minimizeEnergy(maxIterations=200)
	except Exception:
		pass
	n_eq = int(MD_EQUIL_PS * 1000 / MD_TIMESTEP_FS)
	sim.context.setVelocitiesToTemperature(T_K * mmunit.kelvin)
	t0 = time.time()
	chi_samples = []
	save_every = int(MD_FRAME_SAVE_PS * 1000 / MD_TIMESTEP_FS)
	for rep in range(md_replicates):
		sim.step(n_eq)
		n_prod = int(md_ns * 1e6 / MD_TIMESTEP_FS)
		for s in range(0, n_prod, save_every):
			sim.step(save_every)
			state = sim.context.getState(
				getPositions=True, enforcePeriodicBox=False)
			P_nm = np.asarray(
				state.getPositions().value_in_unit(mmunit.nanometer))
			P_A = P_nm * 10.0
			chis = [_dihedral_deg(P_A, *ax) for ax in chi_atom_indices]
			if not any(math.isnan(c) for c in chis):
				chi_samples.append(chis)
	return {
		'chi_samples': np.array(chi_samples, dtype=np.float64),
		'md_seconds':  time.time() - t0,
	}


def refine_well_populations_with_md(grid, capped_rd_mol,
		ase_atoms_template, model_name, phi_atoms, psi_atoms,
		chi_atom_indices, md_ns, md_replicates, top_wells_md,
		platform_name, log):
	log.info(f'Stage 4: MD validation -- {md_ns} ns x {md_replicates} '
		f'rep per top-{top_wells_md} well per bin')
	n_total_md = sum(min(top_wells_md, len(rec.get('wells') or []))
		for rec in grid.values())
	log.info(f'  total MD runs: {n_total_md}')
	t0 = time.time()
	n_done = 0
	for (i, j), rec in grid.items():
		wells = rec.get('wells') or []
		if not wells:
			continue
		phi_deg = rec['phi']; psi_deg = rec['psi']
		for k, w in enumerate(wells[:top_wells_md]):
			ase_at_well = ase_atoms_template.copy()
			try:
				md_res = md_validate_one_well(
					capped_rd_mol, ase_at_well, model_name,
					phi_atoms, psi_atoms, phi_deg, psi_deg,
					chi_atom_indices, md_ns, md_replicates,
					platform_name, log)
				w['md_chi_samples'] = md_res['chi_samples']
			except Exception as e:
				log.warning(f'  MD failed at bin ({i},{j}) well {k}: '
					f'{e}')
				w['md_chi_samples'] = np.zeros(
					(0, len(chi_atom_indices)),
					dtype=np.float64)
			n_done += 1
			if n_done % 25 == 0:
				el = time.time() - t0
				eta = el / n_done * (n_total_md - n_done)
				log.info(f'  MD progress {n_done}/{n_total_md} '
					f'({100*n_done/n_total_md:.1f}%); '
					f'elapsed {el/60:.1f} min; '
					f'ETA {eta/60:.1f} min')
	n_with_md = 0
	for (i, j), rec in grid.items():
		wells = rec.get('wells') or []
		if not wells:
			continue
		for w in wells[:top_wells_md]:
			samples = w.get('md_chi_samples')
			if samples is None or len(samples) == 0:
				continue
			in_basin = sum(1 for sam in samples
				if _chi_dist_linf(sam, w['chi']) <= CHI_CLUSTER_DEG)
			p_md = (in_basin / len(samples)
				if len(samples) > 0 else 0.0)
			w['md_basin_fraction'] = float(p_md)
			n_with_md += 1
		weights = []
		for w in wells:
			weights.append(w.get('md_basin_fraction',
				w.get('prob', 0.0)))
		s = sum(weights) or 1.0
		weights = [x / s for x in weights]
		for w, ww in zip(wells, weights):
			w['prob'] = float(ww)
		wells = [w for w in wells if w['prob'] >= WELL_MIN_PROB]
		s = sum(w['prob'] for w in wells) or 1.0
		for w in wells:
			w['prob'] = w['prob'] / s
		wells.sort(key=lambda w: -w['prob'])
		rec['wells'] = wells
	log.info(f'  MD-refined populations on {n_with_md} wells')


def pipeline_md(cif, tricode, out_path, log):
	if not HAVE_MD:
		raise RuntimeError(
			f'--md requires openmm + openmmml: {_MD_ERR}')
	t0 = time.time()
	log.info('=' * 60)
	log.info(f'Tier 2 (--md) pipeline: cif={cif}, tricode={tricode}, '
		f'model={NN_MODEL}')
	capped, label_to_idx, ace_C, nme_N = (
		parse_cif_and_build_tripeptide(cif, tricode, log))
	chi_axes_names = auto_detect_chi_axes(capped, label_to_idx, log)
	n_chi = len(chi_axes_names)
	chi_atom_indices = resolve_chi_axes(label_to_idx, chi_axes_names)
	phi_atoms, psi_atoms = resolve_phi_psi_atoms(
		label_to_idx, ace_C, nme_N)
	ase_atoms = rdkit_to_ase(capped)
	atoms_template_dict = {
		'symbols':   list(ase_atoms.get_chemical_symbols()),
		'positions': ase_atoms.positions.tolist(),
	}
	log.info(f'Tripeptide ASE: {len(ase_atoms)} atoms; n_chi={n_chi}')
	log.info('Stage 2: NN-potential constrained chi scan')
	grid = run_grid(atoms_template_dict, NN_MODEL,
		phi_atoms, psi_atoms, chi_atom_indices,
		n_chi, n_shared_chi=0, parent_top_chi_grid=None,
		n_workers=WORKERS_DENOVO, log=log)
	fill_empty_bins_with_global(
		grid, atoms_template_dict, NN_MODEL,
		chi_atom_indices, n_chi, n_shared_chi=0, log=log)
	log.info('Stage 3: Hessian-based chi sigmas (representative bin)')
	base_calc = make_nn_calculator(NN_MODEL)
	rep_i, rep_j = 12, 13
	rep_rec = grid.get((rep_i, rep_j))
	rep_sigmas = None
	if rep_rec and rep_rec.get('wells'):
		atoms = Atoms(symbols=atoms_template_dict['symbols'],
			positions=np.array(atoms_template_dict['positions']))
		atoms.calc = base_calc
		try:
			rep_sigmas = chi_sigmas_from_hessian(
				atoms, base_calc, chi_atom_indices, log=log)
			log.info(f'  representative-bin sigmas: '
				f'{[round(s, 1) for s in rep_sigmas]}')
		except Exception as e:
			log.warning(f'  Hessian failed: {e}; using floor sigmas')
	if rep_sigmas is None:
		rep_sigmas = [SIGMA_FLOOR_DEG] * n_chi
	for (i, j), rec in grid.items():
		for w in rec.get('wells') or []:
			w['sigma'] = list(rep_sigmas)
	refine_well_populations_with_md(
		grid, capped, ase_atoms, NN_MODEL,
		phi_atoms, psi_atoms, chi_atom_indices,
		MD_NS_PER_BIN, MD_REPLICATES,
		MD_TOP_WELLS, MD_PLATFORM, log)
	method_extra = {
		'tier':                '2',
		'kind':                f'NN-potential ({NN_MODEL}) scan + '
			f'Hessian sigmas + explicit-water MD per top well',
		'cif':                 os.path.basename(cif),
		'md_ns_per_bin':       float(MD_NS_PER_BIN),
		'md_replicates':       int(MD_REPLICATES),
		'top_wells_md':        int(MD_TOP_WELLS),
		'sigmas_from_hessian': True,
		'representative_bin_sigmas':
			[round(s, 4) for s in rep_sigmas],
		'mlpotential_md':      True,
		'citations': [
			'Devereux et al., JCTC 2020 (ANI-2x)',
			'Eastman et al., JCTC 2024 (OpenMM 8)',
			'Galvelis et al., JCTC 2023 (openmm-ml)',
		],
	}
	emit_rot_v1(grid, n_chi, tricode, chi_axes_names, NN_MODEL,
		method_extra, out_path, log)
	log.info(f'TOTAL wall-time: {(time.time() - t0)/60:.1f} min')


# ----------------------------------------------------------------------
# Tier 1 (--dft) -- RESP + DFT + TIP4P-Ew MD + RR-HO free energy.
# Implemented inside a guard so the heavy import cost is only paid when
# the user explicitly selects --dft.
# ----------------------------------------------------------------------

def _pcm_block():
	# Polarisable Continuum Model spec for Psi4 -- water dielectric.
	return ('\n\tUnits = Angstrom'
		'\n\tMedium {'
		f'\n\t\tSolverType = IEFPCM'
		f'\n\t\tSolvent = {PCM_SOLVENT}'
		'\n\t}'
		'\n\tCavity {'
		'\n\t\tType = GePol'
		'\n\t\tArea = 0.3'
		'\n\t\tMode = Implicit'
		'\n\t}\n')


def _dft_get_backbone(rd_mol):
	out = {'N': None, 'CA': None, 'C': None, 'O': None}
	for k, a in enumerate(rd_mol.GetAtoms()):
		lab = a.GetPropsAsDict().get('cif_label', '')
		if lab in out:
			out[lab] = k
	return out['N'], out['CA'], out['C'], out['O']


def _dft_get_caps(rd_mol):
	ace_C = nme_N = None
	i_N, i_CA, i_C, _ = _dft_get_backbone(rd_mol)
	for nb in rd_mol.GetAtomWithIdx(i_N).GetNeighbors():
		if (nb.GetSymbol() == 'C' and nb.GetIdx() != i_CA
				and not nb.GetPropsAsDict().get('cif_label')):
			ace_C = nb.GetIdx()
			break
	for nb in rd_mol.GetAtomWithIdx(i_C).GetNeighbors():
		if (nb.GetSymbol() == 'N'
				and not nb.GetPropsAsDict().get('cif_label')):
			nme_N = nb.GetIdx()
			break
	return ace_C, nme_N


def _dft_psi4_geometry(rd_mol):
	conf = rd_mol.GetConformer()
	fc = sum(a.GetFormalCharge() for a in rd_mol.GetAtoms())
	lines = [f'{fc} 1']
	for k, a in enumerate(rd_mol.GetAtoms()):
		p = conf.GetAtomPosition(k)
		lines.append(f'{a.GetSymbol()} {p.x:.6f} {p.y:.6f} '
			f'{p.z:.6f}')
	lines.append('units angstrom')
	return '\n'.join(lines)


def _dft_ff_energy(system, rd_mol, conf_id):
	integrator = openmm.LangevinIntegrator(
		300 * mmunit.kelvin, 1 / mmunit.picosecond,
		1 * mmunit.femtosecond)
	ctx = openmm.Context(system, integrator)
	conf = rd_mol.GetConformer(conf_id)
	positions = []
	for k in range(rd_mol.GetNumAtoms()):
		p = conf.GetAtomPosition(k)
		positions.append((p.x * 0.1, p.y * 0.1, p.z * 0.1))
	ctx.setPositions(positions * mmunit.nanometer)
	state = ctx.getState(getEnergy=True)
	return state.getPotentialEnergy().value_in_unit(
		mmunit.kilocalorie_per_mole)


def _dft_qm_single_point(rd_mol, conf_id, functional, basis, pcm=True):
	conf = rd_mol.GetConformer(conf_id)
	lines = []
	for k, a in enumerate(rd_mol.GetAtoms()):
		p = conf.GetAtomPosition(k)
		lines.append(f'{a.GetSymbol()} {p.x:.6f} {p.y:.6f} {p.z:.6f}')
	fc = sum(a.GetFormalCharge() for a in rd_mol.GetAtoms())
	geom = (f'{fc} 1\n' + '\n'.join(lines) + '\nunits angstrom\n')
	mol = psi4.geometry(geom)
	psi4.set_options({
		'basis': basis,
		'scf_type': 'df',
		'reference': 'rks',
	})
	if pcm:
		psi4.set_options({'pcm': True})
		psi4.pcm_helper(_pcm_block())
	E_h = psi4.energy(functional, molecule=mol)
	return float(E_h) * HARTREE2KCAL


def _dft_chi_dist(a, b):
	d = 0.0
	for x, y in zip(a, b):
		delta = abs(((x - y + 180.0) % 360.0) - 180.0)
		if delta > d:
			d = delta
	return d


def _dft_wilson_b_dihedral(xyz, axis):
	i, j, k, l = axis
	N = len(xyz)
	out = np.zeros(3 * N)
	eps = 1e-4
	for atom_idx in (i, j, k, l):
		for d in range(3):
			p = list(map(list, xyz))
			p[atom_idx][d] += eps
			phi_p = _dihedral_deg(p[i], p[j], p[k], p[l])
			p[atom_idx][d] -= 2 * eps
			phi_m = _dihedral_deg(p[i], p[j], p[k], p[l])
			dphi_dx = math.radians(
				((phi_p - phi_m + 180.0) % 360.0) - 180.0
			) / (2 * eps)
			out[3 * atom_idx + d] = dphi_dx
	return out


def _dft_chi_sigmas_from_hessian(rd_mol, xyz, chi_axes):
	fc = sum(a.GetFormalCharge() for a in rd_mol.GetAtoms())
	lines = [f'{fc} 1']
	for k, a in enumerate(rd_mol.GetAtoms()):
		p = xyz[k]
		lines.append(f'{a.GetSymbol()} {p[0]:.6f} {p[1]:.6f} '
			f'{p[2]:.6f}')
	lines.append('units angstrom')
	mol = psi4.geometry('\n'.join(lines))
	psi4.set_options({
		'basis': DFT_OPT_BASIS,
		'scf_type': 'df',
		'pcm': True,
	})
	psi4.pcm_helper(_pcm_block())
	E, wfn = psi4.frequency(DFT_FUNCTIONAL, molecule=mol,
		return_wfn=True)
	H = np.asarray(wfn.hessian())
	sigmas = []
	for axis in chi_axes:
		B = _dft_wilson_b_dihedral(xyz, axis)
		k_h = float(np.einsum('i,ij,j->', B, H, B))
		k_kcal = k_h * HARTREE2KCAL
		if k_kcal <= 0:
			sigma = 60.0
		else:
			sigma_rad = math.sqrt(KT_KCAL / k_kcal)
			sigma = min(math.degrees(sigma_rad), 60.0)
		sigmas.append(sigma)
	return sigmas


def _dft_constrained_optimize(mol_capped, chi_axes, phi_deg, psi_deg,
		chi_start, log):
	geom = _dft_psi4_geometry(mol_capped)
	i_N, i_CA, i_C, _ = _dft_get_backbone(mol_capped)
	ace_C, nme_N = _dft_get_caps(mol_capped)
	dh = []
	dh.append((ace_C, i_N, i_CA, i_C, phi_deg))
	dh.append((i_N, i_CA, i_C, nme_N, psi_deg))
	for k, axis in enumerate(chi_axes):
		a, b, c, d = axis
		dh.append((a, b, c, d, chi_start[k]))
	try:
		psi4.set_options({
			'basis': DFT_OPT_BASIS,
			'scf_type': 'df',
			'g_convergence': 'gau_tight',
			'pcm': True,
			'frozen_dihedral': '\n'.join(
				f'{a+1} {b+1} {c+1} {d+1}'
				for a, b, c, d, _ in dh),
			'fixed_dihedral': '\n'.join(
				f'{a+1} {b+1} {c+1} {d+1} {ang}'
				for a, b, c, d, ang in dh),
		})
		psi4.pcm_helper(_pcm_block())
		psi4.optimize(DFT_FUNCTIONAL)
		psi4.set_options({
			'frozen_dihedral': '',
			'fixed_dihedral': '\n'.join(
				f'{a+1} {b+1} {c+1} {d+1} {ang}'
				for a, b, c, d, ang in dh[:2]),
		})
		psi4.optimize(DFT_FUNCTIONAL)
		psi4.set_options({'basis': DFT_E_BASIS})
		E_h = psi4.energy(DFT_FUNCTIONAL)
		final_geom = psi4.core.get_active_molecule()
		xyz = []
		for k in range(final_geom.natom()):
			xyz.append((final_geom.x(k), final_geom.y(k),
				final_geom.z(k)))
		chi_opt = [_dihedral_deg(xyz[a], xyz[b], xyz[c], xyz[d])
			for a, b, c, d in chi_axes]
		return {
			'chi_opt': chi_opt,
			'E_kcal':  float(E_h) * HARTREE2KCAL,
			'xyz':     xyz,
		}
	except Exception as e:
		log.warning(f'DFT opt failed at start={chi_start}: {e}')
		return None


def _dft_scan_one_node(mol_capped, chi_axes, n_chi, k_canonical,
		phi_deg, psi_deg, log):
	leading = list(itertools.product(CANONICAL_WELLS_DEG,
		repeat=k_canonical))
	ext = [180.0] * (n_chi - k_canonical)
	starts = [list(s) + ext for s in leading]
	wells = []
	for chi_start in starts:
		rec = _dft_constrained_optimize(mol_capped, chi_axes,
			phi_deg, psi_deg, chi_start, log)
		if rec is None:
			continue
		wells.append(rec)
	if not wells:
		raise RuntimeError(
			f'No DFT optima at phi={phi_deg}, psi={psi_deg}')
	chi_arr = np.array([w['chi_opt'] for w in wells])
	if len(chi_arr) > 1:
		D = np.zeros((len(chi_arr), len(chi_arr)))
		for i in range(len(chi_arr)):
			for j in range(i + 1, len(chi_arr)):
				D[i, j] = D[j, i] = _dft_chi_dist(chi_arr[i],
					chi_arr[j])
		condensed = D[np.triu_indices_from(D, k=1)]
		Z = linkage(condensed, method='single')
		labels = fcluster(Z, t=CHI_CLUSTER_DEG,
			criterion='distance')
	else:
		labels = np.array([1])
	clusters = {}
	for w, lab in zip(wells, labels):
		clusters.setdefault(int(lab), []).append(w)
	merged_wells = []
	for lab, members in clusters.items():
		members.sort(key=lambda r: r['E_kcal'])
		rep = members[0]
		sig_chi = _dft_chi_sigmas_from_hessian(mol_capped, rep['xyz'],
			chi_axes)
		merged_wells.append({
			'chi':    [_round(c, 4) for c in rep['chi_opt']],
			'sigma':  [_round(s, 4) for s in sig_chi],
			'E_kcal': rep['E_kcal'],
			'xyz':    rep['xyz'],
		})
	E = np.array([w['E_kcal'] for w in merged_wells])
	E -= E.min()
	Z = np.exp(-E / KT_KCAL).sum()
	for w, e in zip(merged_wells, E):
		w['prob_dft'] = float(np.exp(-e / KT_KCAL) / Z)
	merged_wells = [w for w in merged_wells
		if w['prob_dft'] >= WELL_MIN_PROB]
	s = sum(w['prob_dft'] for w in merged_wells) or 1.0
	for w in merged_wells:
		w['prob_dft'] = w['prob_dft'] / s
	return {
		'phi': float(phi_deg), 'psi': float(psi_deg),
		'wells': sorted(merged_wells, key=lambda w: -w['prob_dft']),
	}


def _dft_rdkit_positions(rd_mol):
	conf = rd_mol.GetConformer()
	positions = []
	for k in range(rd_mol.GetNumAtoms()):
		p = conf.GetAtomPosition(k)
		positions.append((p.x * 0.1, p.y * 0.1, p.z * 0.1))
	return positions


def _dft_md_validate_one_node(parsed_dft, ff_pack, dft_node, log):
	off_mol = ff_pack['off_mol']
	system = ff_pack['system']
	i_N, i_CA, i_C, _ = parsed_dft['i_backbone']
	ace_C, nme_N = _dft_get_caps(parsed_dft['capped'])
	chi_axes = parsed_dft['chi_axes']
	phi_deg = dft_node['phi']; psi_deg = dft_node['psi']
	topology = off_mol.to_topology().to_openmm()
	positions = _dft_rdkit_positions(parsed_dft['capped'])
	modeller = mmapp.Modeller(topology, positions * mmunit.nanometer)
	ff_water = mmapp.ForceField(DFT_WATER_MODEL_XML,
		'amber14/tip4pew.xml')
	modeller.addSolvent(ff_water,
		padding=DFT_WATER_PADDING_A * mmunit.angstrom,
		ionicStrength=DFT_ION_CONC_M * mmunit.molar,
		model='tip4pew')
	system_full = ff_water.createSystem(modeller.topology,
		nonbondedMethod=mmapp.PME,
		nonbondedCutoff=1.0 * mmunit.nanometer,
		constraints=mmapp.HBonds,
		hydrogenMass=4.0 * mmunit.amu)
	# kcal/mol/rad^2 -> kJ/mol/rad^2 conversion (4.184 * 100).
	k_md = PHI_PSI_RESTRAINT_K_MD_KCAL * 4.184 * 100
	for (a, b, c, d, ang0_deg) in [
			(ace_C, i_N, i_CA, i_C, phi_deg),
			(i_N, i_CA, i_C, nme_N, psi_deg)]:
		f = openmm.CustomTorsionForce('0.5*k*(theta - theta0)^2')
		f.addPerTorsionParameter('k')
		f.addPerTorsionParameter('theta0')
		f.addTorsion(a, b, c, d, [k_md, math.radians(ang0_deg)])
		system_full.addForce(f)
	barostat = openmm.MonteCarloBarostat(
		DFT_MD_PRESSURE_BAR * mmunit.bar,
		DFT_MD_TEMP_K * mmunit.kelvin, 25)
	system_full.addForce(barostat)
	dt = DFT_MD_HMR_STEP_FS * mmunit.femtosecond
	results_per_replicate = []
	for rep in range(DFT_MD_REPLICATES):
		integrator = openmm.LangevinMiddleIntegrator(
			DFT_MD_TEMP_K * mmunit.kelvin,
			DFT_MD_FRICTION_PS / mmunit.picosecond, dt)
		ctx = mmapp.Simulation(modeller.topology, system_full,
			integrator)
		ctx.context.setPositions(modeller.positions)
		ctx.minimizeEnergy()
		ctx.context.setVelocitiesToTemperature(
			DFT_MD_TEMP_K * mmunit.kelvin, rep + 1)
		n_eq = int(DFT_MD_EQUIL_NS * 1000 * 1000 / DFT_MD_HMR_STEP_FS)
		ctx.step(n_eq)
		n_prod = int(DFT_MD_NS_PER_NODE * 1000 * 1000 / DFT_MD_HMR_STEP_FS)
		samples = []
		# Save every 20 ps.
		save_every = 5000
		for s in range(0, n_prod, save_every):
			ctx.step(save_every)
			state = ctx.context.getState(getPositions=True)
			positions_nm = np.asarray(
				state.getPositions().value_in_unit(mmunit.nanometer))
			positions_a = positions_nm * 10.0
			chis = [_dihedral_deg(positions_a[a], positions_a[b],
				positions_a[c], positions_a[d])
				for a, b, c, d in chi_axes]
			samples.append(chis)
		results_per_replicate.append(np.asarray(samples))
	pooled = np.concatenate(results_per_replicate, axis=0)
	well_centers = np.array([w['chi'] for w in dft_node['wells']])
	def _nearest(chi):
		d = np.array([
			max(abs(((chi[k] - well_centers[w][k] + 180.0)
				% 360.0) - 180.0)
				for k in range(len(chi)))
			for w in range(len(well_centers))
		])
		return int(np.argmin(d))
	assigns = np.array([_nearest(c) for c in pooled])
	md_pop = np.zeros(len(well_centers))
	for w in range(len(well_centers)):
		md_pop[w] = float(np.mean(assigns == w))
	return {
		'phi': phi_deg, 'psi': psi_deg,
		'md_population': md_pop.tolist(),
		'n_frames': int(pooled.shape[0]),
	}


def _dft_compute_well_free_energies(parsed_dft, ff_pack, dft_node,
		md_node, log):
	wells = dft_node['wells']
	for w in wells:
		fc = sum(a.GetFormalCharge() for a in
			parsed_dft['capped'].GetAtoms())
		lines = [f'{fc} 1']
		for k, a in enumerate(parsed_dft['capped'].GetAtoms()):
			p = w['xyz'][k]
			lines.append(f'{a.GetSymbol()} {p[0]:.6f} {p[1]:.6f} '
				f'{p[2]:.6f}')
		lines.append('units angstrom')
		mol = psi4.geometry('\n'.join(lines))
		psi4.set_options({
			'basis': DFT_OPT_BASIS,
			'pcm':   True,
			't':     T_K,
		})
		psi4.pcm_helper(_pcm_block())
		E, wfn = psi4.frequency(DFT_FUNCTIONAL, molecule=mol,
			return_wfn=True)
		zpe_h = float(psi4.variable('ZPVE'))
		s_vib = float(psi4.variable('THERMAL VIBRATIONAL ENTROPY'))
		pop_md = md_node['md_population']
		idx = wells.index(w)
		p_md = pop_md[idx] if idx < len(pop_md) else 0.0
		p_dft = w['prob_dft']
		if p_md > 0 and p_dft > 0:
			dG_solv_kcal = -KT_KCAL * math.log(p_md / p_dft)
		else:
			dG_solv_kcal = 0.0
		A_kcal = (w['E_kcal']
			+ zpe_h * HARTREE2KCAL
			- T_K * s_vib / 1000.0
			+ dG_solv_kcal)
		w['A_kcal']   = A_kcal
		w['ZPE_kcal'] = zpe_h * HARTREE2KCAL
		w['Svib_e_u'] = s_vib
		w['dG_solv']  = dG_solv_kcal
	A = np.array([w['A_kcal'] for w in wells])
	A -= A.min()
	Z = np.exp(-A / KT_KCAL).sum()
	for w, a in zip(wells, A):
		w['prob_final'] = float(np.exp(-a / KT_KCAL) / Z)
	return wells


def _dft_parse_and_cap(cif_path, tricode, log):
	# Tier 1 needs the chi-axis list as INDEX tuples (not labels), so we
	# wrap parse_cif_and_build_tripeptide and add the canonical chi-chain
	# fast-path that the original NCAA_Rotamers_DFT.py used.
	capped, label_to_idx, ace_C, nme_N = (
		parse_cif_and_build_tripeptide(cif_path, tricode, log))
	residue = tricode.upper()
	if residue in _CHI_CHAINS_BY_RESIDUE:
		chi_axes_labels = _CHI_CHAINS_BY_RESIDUE[residue]
		chi_axes = [tuple(label_to_idx[l] for l in axis)
			for axis in chi_axes_labels]
	else:
		chi_axes_names = auto_detect_chi_axes(capped, label_to_idx, log)
		chi_axes = [tuple(label_to_idx[l] for l in axis)
			for axis in chi_axes_names]
	n_chi = len(chi_axes)
	i_N = label_to_idx['N']; i_CA = label_to_idx['CA']
	i_C = label_to_idx['C']; i_O = label_to_idx['O']
	return {
		'tricode':    residue,
		'capped':     capped,
		'i_backbone': (i_N, i_CA, i_C, i_O),
		'chi_axes':   chi_axes,
		'n_chi':      n_chi,
		'cap':        (ace_C, nme_N),
		'label_to_idx': label_to_idx,
	}


def _dft_build_force_field(parsed_dft, log):
	capped = parsed_dft['capped']
	off_mol = OFFMolecule.from_rdkit(capped, allow_undefined_stereo=True)
	log.info('Stage 1: HF/6-31G(d) ESP for RESP charges')
	grid = LatticeGridSettings(spacing=0.5, inner_vdw_scale=1.4,
		outer_vdw_scale=2.0)
	esp = Psi4ESPSettings(method='hf', basis=RESP_BASIS,
		pcm=True, solvent=PCM_SOLVENT)
	resp_charges = generate_resp_charges(
		[off_mol], esp_settings=esp, grid_settings=grid)
	off_mol.partial_charges = resp_charges * offunit.elementary_charge
	ff = OFFForceField(OPENFF_OFFXML)
	system = ff.create_openmm_system(off_mol.to_topology())
	log.info('Stage 1: FF<->QM gate over 200 random conformations')
	test_mol = Chem.Mol(capped)
	AllChem.EmbedMultipleConfs(test_mol, numConfs=200,
		randomSeed=20260430, pruneRmsThresh=0.5)
	ff_E = []
	qm_E = []
	for conf_id in range(test_mol.GetNumConformers()):
		ff_E.append(_dft_ff_energy(system, test_mol, conf_id))
		qm_E.append(_dft_qm_single_point(test_mol, conf_id,
			DFT_FUNCTIONAL, DFT_E_BASIS))
	ff_E = np.array(ff_E); qm_E = np.array(qm_E)
	ff_E -= ff_E.min(); qm_E -= qm_E.min()
	rmse = float(np.sqrt(np.mean((ff_E - qm_E) ** 2)))
	corr = float(np.corrcoef(ff_E, qm_E)[0, 1])
	log.info(f'  FF<->QM gate: RMSE={rmse:.3f} kcal/mol, r={corr:.3f}')
	if rmse > 1.0 or corr < 0.95:
		raise RuntimeError(
			f'FF/QM gate failed (RMSE={rmse:.3f}, r={corr:.3f}). '
			f'Refusing to proceed with possibly-bad parameters.')
	return {
		'off_mol': off_mol,
		'system':  system,
		'resp_q':  resp_charges,
		'gate':    {'rmse_kcal': rmse, 'r': corr},
	}


def _dft_emit(parsed_dft, results_per_node, method_meta, out_path,
		log):
	# Tier 1's results_per_node uses 'prob_final'; rebuild a grid in the
	# common emit_rot_v1 shape.
	grid = {}
	for rec in results_per_node:
		i = _bin_index(rec['phi'], PHI_START, PHI_STEP, PHI_N)
		j = _bin_index(rec['psi'], PSI_START, PSI_STEP, PSI_N)
		wells = []
		for w in rec['wells']:
			wells.append({
				'chi':   list(w['chi']),
				'sigma': list(w['sigma']),
				'prob':  float(w.get('prob_final',
					w.get('prob_dft', 0.0))),
			})
		grid[(i, j)] = {
			'phi': rec['phi'], 'psi': rec['psi'],
			'wells': wells,
		}
	chi_axes_names_for_method = []
	for axis_idx in parsed_dft['chi_axes']:
		chi_axes_names_for_method.append([
			parsed_dft['capped'].GetAtomWithIdx(a).GetPropsAsDict().get(
				'cif_label', f'atom_{a}')
			for a in axis_idx])
	emit_rot_v1(grid, parsed_dft['n_chi'], parsed_dft['tricode'],
		chi_axes_names_for_method, 'wb97x-d/aug-cc-pVTZ',
		method_meta, out_path, log)


def pipeline_dft(cif, tricode, out_path, log):
	if not HAVE_DFT:
		raise RuntimeError(
			f'--dft requires psi4 + openff-toolkit + openff-recharge + '
			f'mdtraj: {_DFT_ERR}\nInstall via conda for best '
			f'reliability:\n  conda create -n ncaarotamers -c conda-forge '
			f'psi4 openff-toolkit openff-recharge openff-units mdtraj\n'
			f'  pip install -r requirements.txt')
	t0 = time.time()
	log.info('=' * 60)
	log.info(f'Tier 1 (--dft) pipeline: cif={cif}, tricode={tricode}')
	log.info('Stage 0: parse + cap CIF')
	parsed_dft = _dft_parse_and_cap(cif, tricode, log)
	residue = parsed_dft['tricode']
	n_chi = parsed_dft['n_chi']
	if n_chi == 0:
		raise ValueError(
			f'{residue} has no rotatable chi axes; nothing to do.')
	# All chi axes treated as canonical wells for the DFT seed grid.
	k_canonical = n_chi
	log.info(f'  tricode={residue}, n_chi={n_chi}')
	ff_pack = _dft_build_force_field(parsed_dft, log)
	log.info('Stage 2: DFT relaxed scan over 36x36 (phi, psi) grid')
	nodes = _phi_psi_grid()
	dft_results = [None] * len(nodes)
	with ProcessPoolExecutor(max_workers=WORKERS_DFT) as ex:
		futs = {
			ex.submit(_dft_scan_one_node, parsed_dft['capped'],
				parsed_dft['chi_axes'], n_chi, k_canonical,
				phi, psi, log): k
			for k, (i, j, phi, psi) in enumerate(nodes)
		}
		for fut in as_completed(futs):
			k = futs[fut]
			dft_results[k] = fut.result()
			done = sum(1 for r in dft_results if r is not None)
			if done % 50 == 0:
				log.info(f'  DFT progress: {done}/{len(nodes)}')
	log.info('Stage 3: MD validation in TIP4P-Ew water')
	md_results = [None] * len(nodes)
	with ProcessPoolExecutor(max_workers=WORKERS_DFT) as ex:
		futs = {
			ex.submit(_dft_md_validate_one_node, parsed_dft, ff_pack,
				dft_results[k], log): k
			for k in range(len(nodes))
		}
		for fut in as_completed(futs):
			k = futs[fut]
			md_results[k] = fut.result()
	log.info('Stage 4: free-energy decomposition '
		'(A = E + ZPE + S_vib + dG_solv)')
	final_results = []
	max_dP = 0.0
	for dft_node, md_node in zip(dft_results, md_results):
		final = _dft_compute_well_free_energies(parsed_dft, ff_pack,
			dft_node, md_node, log)
		for w_idx, w in enumerate(final):
			pmd = (md_node['md_population'][w_idx]
				if w_idx < len(md_node['md_population']) else 0.0)
			dP = abs(pmd - w['prob_dft'])
			if dP > max_dP:
				max_dP = dP
		final_results.append({
			'phi': dft_node['phi'], 'psi': dft_node['psi'],
			'wells': [{
				'chi':        w['chi'],
				'sigma':      w['sigma'],
				'prob_final': w['prob_final'],
			} for w in final],
		})
	log.info(f'  MD-DFT max dP across all wells: {max_dP:.3f}')
	method_meta = {
		'tier':        '1',
		'kind':        'DFT + explicit-water MD',
		'dft':         f'{DFT_FUNCTIONAL}/{DFT_E_BASIS} // '
			f'{DFT_OPT_BASIS}, PCM(water)',
		'md':          f'OpenFF + RESP/HF-{RESP_BASIS}, TIP4P-Ew, '
			f'{DFT_MD_NS_PER_NODE:.0f} ns x {DFT_MD_REPLICATES} reps/node',
		'free_energy': 'A = E_DFT + ZPE + S_vib(harmonic, RR-HO) + '
			'dG_solv(MD basin)',
		'ff_qm_gate':  ff_pack['gate'],
		'validation_md_dft_max_dP': round(max_dP, 4),
		'citations': [
			'Mardirossian & Head-Gordon, PCCP 2014 (omega-B97X-V)',
			'Smith et al., JCP 2020 (Psi4)',
			'Wagner et al., JCTC 2024 (OpenFF)',
			'Eastman et al., JCTC 2024 (OpenMM 8)',
			'Bayly et al., JPC 1993 (RESP)',
			'Marenich et al., JPCB 2009 (PCM)',
		],
	}
	_dft_emit(parsed_dft, final_results, method_meta, out_path, log)
	log.info(f'TOTAL wall-time: {(time.time() - t0)/3600:.2f} h')


# ----------------------------------------------------------------------
# Main entry-point
# ----------------------------------------------------------------------

def main():
	ap = argparse.ArgumentParser(
		prog='ncaarotamers',
		description='Backbone-dependent rotamer libraries for '
			'non-canonical amino acids. See README.md for methodology.')
	ap.add_argument('--cif', required=True,
		help='RCSB CCD CIF for the residue')
	ap.add_argument('--tricode', required=True,
		help='Three-letter residue code (e.g. ALY)')
	pipe = ap.add_mutually_exclusive_group(required=True)
	pipe.add_argument('--dft', action='store_true',
		help='Tier 1 (HPC, 1-3 weeks): RESP + DFT + MD')
	pipe.add_argument('--md', action='store_true',
		help='Tier 2 (1-4 GPUs, 1-3 days): NN-pot + Hessian + MD')
	pipe.add_argument('--denovo', action='store_true',
		help='Tier 3 (laptop, minutes-hours): NN-pot scan, gas-phase')
	args = ap.parse_args()
	log = setup_logging('ncaarotamers')
	tricode = args.tricode.upper()
	out_dir = Path(__file__).resolve().parent / 'output'
	out_dir.mkdir(exist_ok=True)
	out_path = str(out_dir / f'{tricode}.json')
	if args.dft:
		pipeline_dft(args.cif, tricode, out_path, log)
	elif args.md:
		pipeline_md(args.cif, tricode, out_path, log)
	else:
		pipeline_denovo(args.cif, tricode, out_path, log)


if __name__ == '__main__':
	main()

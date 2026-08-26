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
# Used by the Rosetta MakeRotLib pipeline merged in below.
import collections
import glob
import hashlib
import shutil
import subprocess
import tempfile
import itertools
import json
import logging
import math
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# The neural-network stack is needed by --dft, --md and --denovo but
# not by --rosetta or --swiss, which reach Rosetta and SwissSidechain
# respectively. Guard it so a PyRosetta-only install can still run the
# pipelines it does have, the same way HAVE_MD guards OpenMM below.
try:
	import numpy as np
	import gemmi
	from rdkit import Chem
	from rdkit.Chem import AllChem
	from ase import Atoms
	from ase.optimize import LBFGS
	from ase.calculators.calculator import Calculator, all_changes
	from torchani.models import ANI2x
	HAVE_NN = True
except ImportError as _nn_err:
	HAVE_NN = False
	NN_ERR = str(_nn_err)
	class Calculator: pass
	all_changes = []

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
	from openff.recharge.charges.resp import (
		generate_resp_charge_parameter)
	from openff.recharge.charges.library import (
		LibraryChargeCollection, LibraryChargeGenerator)
	from openff.recharge.esp import ESPSettings, PCMSettings
	from openff.recharge.esp.psi4 import Psi4ESPGenerator
	from openff.recharge.esp.storage import MoleculeESPRecord
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
# Conformers averaged over in the RESP fit. The published protocol
# fits a single conformer; more costs one Psi4 ESP each.
RESP_N_CONFORMERS = 1
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
		#[f'r{k+1}' for k in range(n_chi)] +
		['count', 'prob']
		+ [f'chi{k+1}' for k in range(n_chi)]
		+ [f'sig{k+1}' for k in range(n_chi)])
	bins = [[] for _ in range(PHI_N * PSI_N)]
	top_chi = [[None] * PSI_N for _ in range(PHI_N)]
	for (i, j), rec in grid.items():
		wells = rec.get('wells') or []
		wells_sorted = sorted(wells, key=lambda w: -w['prob'])
		for w in wells_sorted:
			row = []
			#for k in range(n_chi):
			#	row.append(int(_state_index(float(w['chi'][k]))))
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
	# openff-recharge moved this API. The single-call
	# generate_resp_charges(molecules, esp_settings, grid_settings) was
	# replaced by an explicit three step flow: generate conformers,
	# compute an ESP record per conformer with Psi4, then fit RESP
	# charges across those records. Psi4ESPSettings is gone as well,
	# folded into ESPSettings, which now carries the grid and PCM
	# settings rather than taking them as separate arguments.
	grid = LatticeGridSettings(spacing=0.5, inner_vdw_scale=1.4,
		outer_vdw_scale=2.0)
	esp = ESPSettings(method='hf', basis=RESP_BASIS,
		grid_settings=grid,
		pcm_settings=PCMSettings(solvent=PCM_SOLVENT.capitalize()))
	# recharge's own ConformerGenerator is not usable here: its only
	# methods are omega and omega-elf10, both of which need a licensed
	# OpenEye toolkit. The capped residue already carries an RDKit
	# conformer, and from_rdkit preserves it as the pint Quantity that
	# Psi4ESPGenerator expects, so use that directly.
	conformers = list(off_mol.conformers or [])[:RESP_N_CONFORMERS]
	if not conformers:
		raise RuntimeError('capped residue carries no conformer for '
			'the RESP fit')
	log.info('  %d conformer(s) for the RESP fit' % len(conformers))
	records = []
	for conf in conformers:
		out_conf, grid_xyz, esp_vals, field = Psi4ESPGenerator.generate(
			off_mol, conf, esp, minimize=False)
		records.append(MoleculeESPRecord.from_molecule(
			off_mol, out_conf, grid_xyz, esp_vals, field, esp))
	param = generate_resp_charge_parameter(records, solver=None)
	# The fitted values are ordered by the indexed SMILES the parameter
	# carries, not by the molecule's atom order. LibraryChargeGenerator
	# is what maps one onto the other; indexing param.value directly
	# would silently mis-assign charges.
	resp_charges = LibraryChargeGenerator.generate(off_mol,
		LibraryChargeCollection(parameters=[param]))
	off_mol.partial_charges = (np.asarray(resp_charges).flatten()
		* offunit.elementary_charge)
	ff = OFFForceField(OPENFF_OFFXML)
	# Without charge_from_molecules the toolkit ignores the charges just
	# assigned above and re-derives its own with AM1BCC, which both
	# discards the RESP fit this stage exists to produce and fails
	# outright when no AM1BCC-capable toolkit is installed.
	system = ff.create_openmm_system(off_mol.to_topology(),
		charge_from_molecules=[off_mol])
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
			f'mdtraj: {_DFT_ERR}\nThese are conda-forge only and are '
			f'not on PyPI. Build the environment with:\n'
			f'  bash setup.sh')
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


# ======================================================================
# Rosetta MakeRotLib pipeline (--rosetta)
#
# Merged from NCAA_PyRosetta.py. Nothing here is hardcoded per residue:
# atom types, partial charges, ionisation, ring perception, chi axes and
# rotamer wells are all derived from the CIF bond graph, and the atom
# typing reproduces Rosetta's own assignments for the twenty canonical
# amino acids exactly. The scan itself is Rosetta's MakeRotLib (Renfrew
# et al., PLoS ONE 2012, e32637) driven through the job distributor, so
# the minimisation, k-means clustering, Boltzmann populations and well
# widths are Rosetta's rather than a reimplementation.
# ======================================================================

BOND = {'SING': 1, 'DOUB': 2, 'TRIP': 3, 'AROM': 1.5}
ELEMZ = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16,
	'CL': 17, 'SE': 34, 'BR': 35, 'I': 53}
BACKBONE = ('N', 'CA', 'C', 'O')
# The score function, inlined so this script is self contained and a
# clone reproduces the published libraries with no side files.
#
# It is NOT make_rot_lib_orig.wts. CHARMM parameterises a rotatable bond
# as torsion plus 1-4 Lennard-Jones plus 1-4 Coulomb together, and the
# distributed file ships only the repulsive half of that matched pair,
# which systematically over-populates gauche minus chi1. These six
# weights restore the missing channel and take intra-residue sterics
# from Rosetta's etable. Measured cost of the change, as modal rotamer
# accuracy against Dunbrack: SER 26.6 -> 57.0, PHE 55.5 -> 79.0 and
# TYR 52.8 -> 79.5, against THR 87.0 -> 80.3, HIS 62.2 -> 15.9 and
# TRP chi1 65.3 -> 50.9. HIS is a regression caused by this choice.
#
# Rosetta loads weights from a path, so these lines are written to a
# temporary file at startup. Set MRL_WTS to a path to override, which
# is how the weight bisect was run; the emitted provenance hash is
# taken over the parsed weight lines either way, so it identifies the
# numbers rather than the file, and reformatting the source or editing
# these comments does not change it.
WTS_INLINE = """mm_twist         5.0
mm_lj_intra_rep  0.1
mm_lj_intra_atr  1.5
fa_intra_rep     1.0
fa_intra_atr     0.2
fa_intra_elec    0.6
"""


def wtspath():
	"""
	Resolve the score function to a path that Rosetta can load
	Arguments:
	----------
		No arguments taken
	Returns:
	--------
		str: MRL_WTS when set, else a temp file holding WTS_INLINE
	"""
	env = os.environ.get('MRL_WTS')
	if env: return env
	fd, path = tempfile.mkstemp(prefix='mrl_wts_', suffix='.wts')
	os.write(fd, WTS_INLINE.encode())
	os.close(fd)
	return path


def wtshash():
	"""
	Provenance hash over the parsed weight lines, not the file bytes
	Arguments:
	----------
		No arguments taken
	Returns:
	--------
		str: first 16 hex digits of the sha256 of the sorted weights
	"""
	env = os.environ.get('MRL_WTS')
	src = open(env).read() if env else WTS_INLINE
	lines = sorted(' '.join(l.split('#')[0].split())
		for l in src.splitlines() if l.split('#')[0].split())
	return hashlib.sha256('\n'.join(lines).encode()).hexdigest()[:16]


WEIGHTS = wtspath()
# The value used by the run that produced the published libraries, read
# from the protocol capture shipped as Supporting Information S2 of
# Renfrew et al. 2012 ("Kb T value used: 0.60"). It is not fitted here.
# It transfers because the energy scale matches: across the 1755 rotamer
# lines of that log the reported total is the plain sum of its torsion
# and intra-residue LJ channels, so all three weights were unity, which
# is what make_rot_lib_orig.wts sets. MakeRotLib's compiled-in default
# is 1.4 and the current docs template says 1; neither is the paper's.
KBT = 0.60
# Floor for an emitted sigma, in degrees. MakeRotLib reports the half
# width of a 0.5 kcal energy well on an isolated residue; Pose reads
# that column as a conformational spread and scores 0.5*(dchi/sigma)^2
# with it. Those are different quantities and the computed one is much
# the narrower, so an ungoverned value makes a generated residue far
# stiffer than any real one: a 15 degree error costs 2.7 energy units
# against Dunbrack's valine sigma and 44 against the raw computed one.
# 1.3 is the smallest sigma anywhere in Dunbrack's canonical set, so
# nothing generated is scored more stiffly than the stiffest observed
# rotamer. This bounds an artefact; it does not convert one quantity
# into the other, and the semantic difference has to be disclosed.
SIGMIN = 1.3
# Whether to relax bond angles and lengths against CHARMM equilibria
# before the scan. 'auto' relaxes every residue except beta branched
# ones. An idealised CIF block puts every angle at a tetrahedral 109.5,
# which is already right when CB carries two heavy substituents and
# several degrees wrong when it carries one. An earlier version tested
# the CHARMM CA-CB-X equilibrium against 2.0 degrees, which is unsound:
# those equilibria are discrete table entries and every aromatic CB sits
# at exactly 2.00, so the comparison ties and the outcome is decided by
# whether the test is written >= or >. Beta branching is the property
# that angle was standing in for, Rosetta names it directly as
# BETA_BRANCHED_SIDECHAIN, and it cannot tie.
RELAX = 'auto'


def log(msg):
	'''
	Write a progress line to stderr
	Arguments:
	----------
		msg: str - the message to write
	Returns:
	--------
		Nothing, writes to stderr
	'''
	sys.stderr.write(msg + '\n')
	sys.stderr.flush()


def readcif(path, tri):
	'''
	Parse a CCD CIF into atom and bond records keyed by column name
	Arguments:
	----------
		path: str - path to the component CIF file
		tri: str - the three letter component code
	Returns:
	--------
		list: one dict per atom, keys are the CIF column names
		list: one dict per bond, keys are the CIF column names
	'''
	atoms, bonds, acol, bcol, mode = [], [], [], [], None
	for raw in open(path):
		s = raw.strip()
		if s.startswith('_chem_comp_atom.'):
			acol.append(s.split('.', 1)[1]); mode = 'a'; continue
		if s.startswith('_chem_comp_bond.'):
			bcol.append(s.split('.', 1)[1]); mode = 'b'; continue
		if not s or s.startswith(('#', 'loop_', 'data_', '_')): continue
		t = s.split()
		if not t or t[0] != tri: continue
		if mode == 'a' and len(t) == len(acol):
			atoms.append(dict(zip(acol, t)))
		elif mode == 'b' and len(t) == len(bcol):
			bonds.append(dict(zip(bcol, t)))
	return atoms, bonds


def digest(atoms, bonds, prefer=None):
	'''
	Reduce the CIF records to a bond graph over the non leaving atoms
	Arguments:
	----------
		atoms: list - atom dicts from readcif
		bonds: list - bond dicts from readcif
	Returns:
	--------
		dict: per atom element, formal charge, aromatic flag and ordinal
		dict: atom name to set of bonded atom names
		dict: frozenset of two atom names to bond order
		dict: atom name to ideal xyz tuple
		list: kept atom names in CIF order
		str: the CIF coordinate block the geometry was seeded from
	'''
	info, adj, order, xyz = {}, collections.defaultdict(set), {}, {}
	# Pick one coordinate block for the whole component. The two blocks
	# sit in unrelated frames, so falling back per atom can splice a
	# residue together from centroids hundreds of angstrom apart. The
	# idealised block is preferred: a single experimental conformer
	# carries that one crystal's strain, and its provenance is not
	# recorded for most CCD components. Either way the geometry is only
	# a seed, since the internal coordinates are relaxed downstream.
	src = prefer or 'pdbx_model_Cartn_x_ideal'
	if any(a.get(src, '?') in ('?', '.') for a in atoms):
		src = ('model_Cartn_x' if src != 'model_Cartn_x'
			else 'pdbx_model_Cartn_x_ideal')
	sy, sz = src.replace('_x', '_y'), src.replace('_x', '_z')
	for a in atoms:
		n = a['atom_id'].strip('"')
		info[n] = {'el': a['type_symbol'].capitalize(),
			'q': int((a.get('charge', '0') or '0').replace('?', '0')),
			'aro': a.get('pdbx_aromatic_flag', 'N') == 'Y',
			'leave': a.get('pdbx_leaving_atom_flag', 'N') == 'Y',
			'i': int(a.get('pdbx_ordinal', '0') or 0)}
		if a.get(src, '?') not in ('?', '.'):
			xyz[n] = (float(a[src]), float(a[sy]), float(a[sz]))
	keep = [n for n in info if not info[n]['leave']]
	keep.sort(key=lambda n: info[n]['i'])
	for b in bonds:
		x, y = b['atom_id_1'].strip('"'), b['atom_id_2'].strip('"')
		if x not in keep or y not in keep: continue
		adj[x].add(y); adj[y].add(x)
		o = BOND.get(b['value_order'].upper(), 1)
		if b.get('pdbx_aromatic_flag', 'N') == 'Y': o = 1.5
		order[frozenset((x, y))] = o
	return info, adj, order, xyz, keep, src


def bridges(adj, keep, info):
	'''
	Find every bond that does not lie in a ring, by Tarjan bridge finding
	Arguments:
	----------
		adj: dict - atom name to set of bonded atom names
		keep: list - atom names to consider
		info: dict - per atom properties from digest
	Returns:
	--------
		set: frozensets of the two atom names of each ring bond
	'''
	hv = [n for n in keep if info[n]['el'] != 'H']
	hadj = {n: [m for m in adj[n] if m in hv] for n in hv}
	num, low, ring, cnt = {}, {}, set(), [0]
	for root in hv:
		if root in num: continue
		stack = [(root, None, iter(hadj[root]))]
		num[root] = low[root] = cnt[0]; cnt[0] += 1
		while stack:
			u, pu, it = stack[-1]
			nxt = next(it, None)
			if nxt is None:
				stack.pop()
				if stack:
					p = stack[-1][0]
					low[p] = min(low[p], low[u])
					if low[u] > num[p]: continue
					ring.add(frozenset((p, u)))
				continue
			if nxt == pu: continue
			if nxt in num:
				low[u] = min(low[u], num[nxt])
				ring.add(frozenset((u, nxt)))
				continue
			num[nxt] = low[nxt] = cnt[0]; cnt[0] += 1
			stack.append((nxt, u, iter(hadj[nxt])))
	return ring


def ringsizes(adj, ring):
	'''
	Smallest ring size through each atom that lies in a ring
	Arguments:
	----------
		adj: dict - atom name to set of bonded atom names
		ring: set - ring bonds as frozensets of two atom names
	Returns:
	--------
		dict: atom name to smallest ring size, absent when not in a ring
	'''
	out = {}
	for n in adj:
		nbr = [m for m in adj[n] if frozenset((n, m)) in ring]
		best = 0
		for s in nbr:
			seen = {n: 0, s: 1}
			q = [s]
			while q:
				u = q.pop(0)
				for v in adj[u]:
					if v == n and u != s:
						best = min(best or 99, seen[u] + 1)
						continue
					if v in seen or frozenset((u, v)) not in ring: continue
					seen[v] = seen[u] + 1
					q.append(v)
		if best: out[n] = best
	return out


def protonate(info, adj, order, keep, xyz):
	'''
	Move the residue to its physiological ionisation, because the CCD
	ships neutral forms while Rosetta uses charged ones
	Arguments:
	----------
		info: dict - per atom properties from digest
		adj: dict - atom name to set of bonded atom names
		order: dict - frozenset of two atom names to bond order
		keep: list - kept atom names in CIF order
		xyz: dict - atom name to ideal xyz tuple, adjusted in place
	Returns:
	--------
		list: the kept atom names after removing acidic hydrogens
		list: the names of the hydrogens that were removed
	'''
	drop = []
	for n in list(keep):
		if info[n]['el'] != 'O': continue
		h = [m for m in adj[n] if info[m]['el'] == 'H']
		hv = [m for m in adj[n] if info[m]['el'] != 'H']
		if not h or not hv: continue
		p = hv[0]
		if info[p]['el'] not in ('C', 'P', 'S'): continue
		nox = [m for m in adj[p] if info[m]['el'] == 'O']
		if info[p]['el'] == 'C' and len(nox) < 2: continue
		drop.extend(h)
		info[n]['q'] = -1
	for n in list(keep):
		if info[n]['el'] != 'N' or not info[n]['aro']: continue
		if not any(info[m]['el'] == 'H' for m in adj[n]): continue
		ring2 = [m for m in adj[n] if info[m]['el'] == 'C'
			and info[m]['aro']]
		twin = [x for m in ring2 for x in adj[m]
			if x != n and info[x]['el'] == 'N' and info[x]['aro']
			and any(info[y]['el'] == 'H' for y in adj[x])]
		if not twin: continue
		near = [m for m in ring2 if any(info[x]['el'] != 'H'
			and not info[x]['aro'] for x in adj[m])]
		if near:
			drop.extend(m for m in adj[n] if info[m]['el'] == 'H')
			info[n]['q'] = 0
			break
	if not drop: return keep, []
	for d in drop:
		for m in list(adj[d]):
			adj[m].discard(d)
			order.pop(frozenset((d, m)), None)
		adj.pop(d, None)
	keep = [n for n in keep if n not in drop]
	# A deprotonated acid is symmetric, but the CIF geometry is not: the
	# former hydroxyl still carries its longer single-bond distance. Even
	# the terminal oxygens of each centre so the sterics are those of the
	# real ion rather than of the neutral acid with its proton deleted.
	for p in keep:
		if info[p]['el'] not in ('C', 'P', 'S'): continue
		term = [m for m in adj[p] if info[m]['el'] == 'O'
			and len([x for x in adj[m] if info[x]['el'] != 'H']) == 1
			and not [x for x in adj[m] if info[x]['el'] == 'H']]
		if len(term) < 2 or p not in xyz: continue
		d = [math.dist(xyz[m], xyz[p]) for m in term if m in xyz]
		if len(d) < 2: continue
		avg = sum(d) / len(d)
		for m in term:
			if m not in xyz: continue
			v = [xyz[m][i] - xyz[p][i] for i in range(3)]
			ln = math.sqrt(sum(x * x for x in v)) or 1.0
			xyz[m] = tuple(xyz[p][i] + v[i] * avg / ln for i in range(3))
	return keep, drop


def alpha(n, info, adj, order):
	'''
	True when the atom is the alpha carbon, bonded to a nitrogen and to a
	carbonyl carbon
	Arguments:
	----------
		n: str - the atom name being tested
		info: dict - per atom properties from digest
		adj: dict - atom name to set of bonded atom names
		order: dict - frozenset of two atom names to bond order
	Returns:
	--------
		bool: True when the atom looks like an alpha carbon
	'''
	if info[n]['el'] != 'C': return False
	nb = [m for m in adj[n] if info[m]['el'] != 'H']
	if not any(info[m]['el'] == 'N' for m in nb): return False
	return any(info[m]['el'] == 'C' and any(info[x]['el'] == 'O'
		for x in adj[m] if order.get(frozenset((m, x)), 1) >= 2)
		for m in nb)


def carbonkind(n, info, adj, order):
	'''
	Classify a carbon by the heteroatoms it carries, to separate amide,
	carboxylate and guanidinium centres
	Arguments:
	----------
		n: str - the carbon being classified
		info: dict - per atom properties from digest
		adj: dict - atom name to set of bonded atom names
		order: dict - frozenset of two atom names to bond order
	Returns:
	--------
		str: one of guanidinium, carboxyl, amide or plain
	'''
	nb = [m for m in adj[n] if info[m]['el'] != 'H']
	nn = sum(1 for m in nb if info[m]['el'] == 'N')
	no = sum(1 for m in nb if info[m]['el'] == 'O')
	if nn >= 3: return 'guanidinium'
	# A carboxylate needs a double bond, not merely two oxygens. An
	# acetal or a methyl ester also carries two, and typing those OC/COO
	# invents a formal negative charge on a neutral group. Never fires
	# on a canonical; on a novel component it would ship a normal
	# looking library built on an anion that is not there.
	dbl = any(order.get(frozenset((n, m)), 1) >= 2
		for m in adj[n] if info[m]['el'] == 'O')
	if no >= 2 and dbl: return 'carboxyl'
	if no >= 1 and nn >= 1: return 'amide'
	return 'plain'


def mmtype(n, info, adj, order, ring, rs, fused):
	'''
	Assign a CHARMM molecular mechanics atom type from the bond graph
	Arguments:
	----------
		n: str - the atom being typed
		info: dict - per atom properties from digest
		adj: dict - atom name to set of bonded atom names
		order: dict - frozenset of two atom names to bond order
		ring: set - ring bonds as frozensets of two atom names
		rs: dict - atom name to smallest ring size
		fused: bool - True when the side chain has a fused ring system
	Returns:
	--------
		str: the name of a CHARMM atom type
	'''
	el = info[n]['el']
	nb = [m for m in adj[n] if info[m]['el'] != 'H']
	cyc = [m for m in adj.get('N', ()) if info.get(m, {}).get('el') == 'C'
		and frozenset(('N', m)) in ring]
	if cyc and n in rs:
		if el == 'N': return 'N'
		if el == 'C':
			if alpha(n, info, adj, order): return 'CP1'
			return 'CP3' if 'N' in adj[n] else 'CP2'
	nh = sum(1 for m in adj[n] if info[m]['el'] == 'H')
	dbl = [m for m in adj[n] if order.get(frozenset((n, m)), 1) >= 2]
	nring = sum(1 for m in adj[n] if frozenset((n, m)) in ring)
	if el == 'H':
		if not nb: return 'H'
		p = nb[0]
		pe = info[p]['el']
		if pe == 'S': return 'HS'
		if pe == 'O': return 'H'
		if pe == 'N':
			pn = [m for m in adj[p] if info[m]['el'] != 'H']
			chg = info[p]['q'] > 0 or any(carbonkind(m, info, adj, order)
				== 'guanidinium' for m in pn if info[m]['el'] == 'C')
			return 'HC' if chg else 'H'
		if info[p]['aro']:
			pn = [m for m in adj[p] if frozenset((p, m)) in ring]
			nn = sum(1 for m in pn if info[m]['el'] == 'N')
			if rs.get(p) == 5 and not fused:
				return 'HR1' if nn >= 2 else 'HR3'
			return 'HP'
		if alpha(p, info, adj, order): return 'HB'
		return 'HA'
	if el == 'C':
		if info[n]['aro']:
			if nring >= 3: return 'CPT'
			if rs.get(n) == 5 and not fused:
				nn = sum(1 for m in adj[n]
					if frozenset((n, m)) in ring and info[m]['el'] == 'N')
				return 'CPH2' if nn >= 2 else 'CPH1'
			if rs.get(n) == 5 and fused:
				return 'CY' if any(frozenset((n, m)) not in ring
					and info[m]['el'] != 'H' for m in adj[n]) else 'CA'
			return 'CA'
		k = carbonkind(n, info, adj, order)
		if k == 'guanidinium': return 'C'
		if k in ('carboxyl', 'amide'):
			return 'C' if any(alpha(m, info, adj, order) for m in nb) \
				else 'CC'
		if dbl: return 'C'
		return {3: 'CT3', 2: 'CT2', 1: 'CT1'}.get(nh, 'CT')
	if el == 'N':
		if info[n]['aro']:
			if fused: return 'NY'
			return 'NR1' if nh else 'NR2'
		if nh >= 3: return 'NH3'
		pn = [m for m in nb if info[m]['el'] == 'C']
		if any(carbonkind(m, info, adj, order) == 'guanidinium'
			for m in pn): return 'NC2'
		if nh == 2: return 'NH2'
		return 'NH1'
	if el == 'O':
		if any(info[m]['el'] == 'P' for m in nb):
			return 'ON2' if len(nb) >= 2 else 'ON3'
		p = nb[0] if nb else None
		k = carbonkind(p, info, adj, order) if p and info[p]['el'] == 'C' \
			else 'plain'
		if k == 'carboxyl': return 'OC'
		if nh: return 'OH1'
		if dbl: return 'O'
		return 'OS'
	if el == 'S': return 'SM' if any(info[m]['el'] == 'S' for m in nb) else 'S'
	if el == 'P': return 'P'
	# CHARMM types the heavy halogens directly. Fluorine is split by the
	# carbon it hangs off, following the CGenFF provenance recorded in
	# mm_atom_properties.txt: FGR1 aromatic, FGA1/2/3 mono/di/tri.
	# The placeholder type X carries zero radius and zero well depth, so
	# falling through to it would hide the atom from mm_lj entirely.
	if el in ('Cl', 'Br', 'I'): return el.upper()
	if el == 'F':
		c = [m for m in nb if info[m]['el'] == 'C']
		if not c: return 'F1'
		if info[c[0]]['aro']: return 'FA'
		f = sum(1 for m in adj[c[0]] if info[m]['el'] == 'F')
		return {1: 'F1', 2: 'F2'}.get(f, 'F3')
	sys.exit('[-] Error: no CHARMM mm atom type for element %s (atom %s); '
		'Rosetta ships none for it and inventing one is not supported'
		% (el, n))


def rosettatype(n, info, adj, order, mm):
	'''
	Assign a Rosetta full atom type, using the CHARMM type already
	derived plus the local bonding as the discriminator
	Arguments:
	----------
		n: str - the atom being typed
		info: dict - per atom properties from digest
		adj: dict - atom name to set of bonded atom names
		order: dict - frozenset of two atom names to bond order
		mm: str - the CHARMM atom type from mmtype
	Returns:
	--------
		str: the name of a Rosetta full atom type
	'''
	el = info[n]['el']
	nh = sum(1 for m in adj[n] if info[m]['el'] == 'H')
	hv = [m for m in adj[n] if info[m]['el'] != 'H']
	bb = n in ('N', 'CA', 'C', 'O')
	if el == 'H':
		if mm == 'HS': return 'HS'
		if mm == 'HP': return 'Haro'
		if mm == 'HC': return 'Hpol'
		if mm in ('HA', 'HB', 'HR1', 'HR3'): return 'Hapo'
		return 'HNbb' if hv and hv[0] == 'N' else 'Hpol'
	if el == 'C':
		if n == 'C': return 'CObb'
		if n == 'CA' or mm == 'CP1': return 'CAbb'
		if mm in ('CP2', 'CP3'): return 'CH2'
		if mm in ('CA', 'CPH1', 'CPH2'): return 'aroC' if nh else 'CH0'
		if mm in ('CPT', 'CY'): return 'CH0'
		if mm == 'C': return 'aroC'
		if mm == 'CC':
			return 'COO' if carbonkind(n, info, adj, order) == 'carboxyl' \
				else 'CNH2'
		return {'CT3': 'CH3', 'CT2': 'CH2', 'CT1': 'CH1'}.get(mm, 'CH0')
	if el == 'N':
		if mm == 'N': return 'Npro'
		if mm == 'NC2':
			return 'NtrR' if len(hv) >= 2 else 'Narg'
		if bb: return 'Nbb'
		return {'NH2': 'NH2O', 'NH3': 'Nlys', 'NR1': 'Ntrp',
			'NR2': 'Nhis', 'NY': 'Ntrp'}.get(mm, 'Nbb')
	if el == 'O':
		if bb: return 'OCbb'
		return {'OC': 'OOC', 'OH1': 'OH', 'O': 'ONH2', 'ON2': 'OH',
			'ON3': 'OOC'}.get(mm, 'OOC')
	if el == 'S': return 'SH1' if nh else 'S'
	if el == 'P': return 'Phos'
	# Rosetta ships real Lennard-Jones parameters for the halogens and
	# selenium, so use them. Never fall through to VIRT: a virtual atom
	# has zero radius and zero well depth, so the scan would silently
	# treat the atom as absent and emit a normal-looking library.
	if el in ('F', 'Cl', 'Br', 'I', 'Se'): return el
	sys.exit('no Rosetta atom type for element %s (atom %s); this residue '
		'cannot be scanned without a virtual atom' % (el, n))


def rotatable(a, b, info, adj, order, ring):
	'''
	Decide whether the bond a-b can be treated as a free single bond
	Arguments:
	----------
		a: str - name of the first atom of the bond
		b: str - name of the second atom of the bond
		info: dict - per atom properties from digest
		adj: dict - atom name to set of bonded atom names
		order: dict - frozenset of two atom names to bond order
		ring: set - ring bonds from bridges
	Returns:
	--------
		bool: True when the bond defines a chi angle
	'''
	if frozenset((a, b)) in ring: return False
	if order.get(frozenset((a, b)), 1) != 1: return False
	hv = lambda n, o: [m for m in adj[n]
		if info.get(m, {}).get('el') != 'H' and m != o]
	if not hv(a, b) or not hv(b, a): return False
	if info[a]['el'] in ('N', 'O') and info[b]['el'] == 'C':
		for m in adj[b]:
			if m != a and info[m]['el'] in ('N', 'O') and \
					order.get(frozenset((b, m)), 1) >= 2:
				return False
	return True


def priority(n, info):
	'''
	Rank a substituent so the reference atom of a chi is chosen the way
	the IUPAC convention does, by element then by locant
	Arguments:
	----------
		n: str - the atom name being ranked
		info: dict - per atom properties from digest
	Returns:
	--------
		tuple: sort key, lowest sorts first
	'''
	d = ''.join(c for c in n if c.isdigit())
	return (-ELEMZ.get(info[n]['el'].upper(), 0),
		int(d) if d else 0, info[n]['i'])


def findchi(info, adj, order, ring):
	'''
	Detect the chi axes of the side chain with ring perception
	Arguments:
	----------
		info: dict - per atom properties from digest
		adj: dict - atom name to set of bonded atom names
		order: dict - frozenset of two atom names to bond order
		ring: set - ring bonds from bridges
	Returns:
	--------
		list: one tuple of four atom names per chi, in chi order
	'''
	hv = lambda n: info.get(n, {}).get('el') != 'H'
	side = {n for n in info if hv(n) and not info[n]['leave']
		and n not in ('N', 'CA', 'C', 'O', 'OXT')}
	if 'CB' not in side: return []
	par, dist, q = {'CA': 'N', 'CB': 'CA'}, {'CB': 0}, ['CB']
	while q:
		u = q.pop(0)
		for v in sorted(adj[u], key=lambda m: info.get(m, {}).get('i', 0)):
			if v not in side: continue
			# A ring closes on an atom reachable from two branches at the
			# same depth. Rosetta's atom tree attaches it to the later
			# branch, so tyrosine gets CZ from CE2 and its third chi
			# reads CE2-CZ-OH-P rather than CE1-CZ-OH-P, which is 180
			# degrees away. Match that or the emitted axes disagree with
			# every other definition of the same residue.
			if v in dist:
				if (dist[v] == dist[u] + 1
						and info[u]['i'] > info[par[v]]['i']):
					par[v] = u
				continue
			dist[v] = dist[u] + 1; par[v] = u; q.append(v)
	chis = []
	for c in sorted(dist, key=lambda m: (dist[m], info[m]['i'])):
		b = par[c]
		if not rotatable(b, c, info, adj, order, ring): continue
		cand = [m for m in adj[c] if m != b and hv(m)
			and not info[m]['leave']]
		if not cand or par.get(b) is None: continue
		chis.append((par[b], b, c, min(cand,
			key=lambda m: priority(m, info))))
	return chis


def classes(info, adj, keep):
	'''
	Colour atoms by iterative neighbourhood refinement so that
	topologically equivalent atoms share a colour
	Arguments:
	----------
		info: dict - per atom properties from digest
		adj: dict - atom name to set of bonded atom names
		keep: list - atom names to colour
	Returns:
	--------
		dict: atom name to an integer colour
	'''
	lab = {n: (info[n]['el'], info[n]['aro']) for n in keep}
	for _ in range(6):
		new = {n: (lab[n], tuple(sorted(str(lab[m]) for m in adj[n]
			if m in lab))) for n in keep}
		uniq = {v: i for i, v in enumerate(sorted(set(new.values()), key=str))}
		lab = {n: uniq[new[n]] for n in keep}
	return lab


def findsym(chis, info, adj, order, ring, keep):
	'''
	Report the rotational period of each chi implied by the bond graph
	Arguments:
	----------
		chis: list - chi axes from findchi
		info: dict - per atom properties from digest
		adj: dict - atom name to set of bonded atom names
		order: dict - frozenset of two atom names to bond order
		ring: set - ring bonds from bridges
		keep: list - atom names to colour
	Returns:
	--------
		list: 360, 180 or 120 for each chi, recorded as provenance only
	'''
	cl = classes(info, adj, keep)
	per = []
	for a, b, c, d in chis:
		sub = [m for m in adj[c] if m != b]
		rng = [m for m in sub if frozenset((c, m)) in ring]
		sp2 = any(order.get(frozenset((c, m)), 1) >= 2 for m in adj[c])
		same = len({cl[m] for m in sub}) == 1
		p = 360
		if len(sub) == 3 and same and not rng: p = 120
		elif len(rng) == 2 and cl[rng[0]] == cl[rng[1]]: p = 180
		elif len(sub) == 2 and same and sp2 and not rng: p = 180
		per.append(p)
	return per


def geometry(xyz, a, b, c, d):
	'''
	Measure a distance, a bond angle and a torsion from ideal coordinates
	Arguments:
	----------
		xyz: dict - atom name to ideal xyz tuple
		a: str - the atom being placed
		b: str - its parent
		c: str - its grandparent
		d: str - its great grandparent
	Returns:
	--------
		tuple: distance in angstroms, 180 minus the angle, and the torsion
	'''
	sub = lambda u, v: tuple(u[i] - v[i] for i in range(3))
	dot = lambda u, v: sum(u[i] * v[i] for i in range(3))
	crs = lambda u, v: (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2],
		u[0]*v[1]-u[1]*v[0])
	nrm = lambda u: math.sqrt(dot(u, u))
	unt = lambda u: tuple(x / nrm(u) for x in u)
	A, B, C, D = xyz[a], xyz[b], xyz[c], xyz[d]
	ang = math.degrees(math.acos(max(-1.0, min(1.0,
		dot(unt(sub(A, B)), unt(sub(C, B)))))))
	n1, n2 = crs(sub(B, A), sub(C, B)), crs(sub(C, B), sub(D, C))
	tor = math.degrees(math.atan2(
		dot(crs(n1, n2), unt(sub(C, B))), dot(n1, n2)))
	return nrm(sub(A, B)), 180.0 - ang, tor


def types(info, adj, order, keep, ring):
	'''
	Derive the Rosetta type, CHARMM type and partial charge of every atom
	Arguments:
	----------
		info: dict - per atom properties from digest
		adj: dict - atom name to set of bonded atom names
		order: dict - frozenset of two atom names to bond order
		keep: list - kept atom names in CIF order
		ring: set - ring bonds as frozensets of two atom names
	Returns:
	--------
		dict: atom name to a tuple of Rosetta type, CHARMM type and charge
	'''
	from pyrosetta.rosetta.core.chemical import (ChemicalManager,
		MutableResidueType, BondName, AA, rosetta_recharge_fullatom)
	rs = ringsizes(adj, ring)
	fused = any(sum(1 for m in adj[n] if frozenset((n, m)) in ring) >= 3
		for n in keep)
	cm = ChemicalManager.get_instance()
	rt = MutableResidueType(cm.atom_type_set('fa_standard'),
		cm.element_set('default'), cm.mm_atom_type_set('fa_standard'),
		cm.orbital_type_set('fa_standard'))
	rt.name('TMP'); rt.name3('TMP'); rt.name1('X'); rt.aa(AA.aa_unk)
	bn = {1: BondName.SingleBond, 2: BondName.DoubleBond,
		3: BondName.TripleBond, 1.5: BondName.AromaticBond}
	out = {}
	for n in keep:
		mm = mmtype(n, info, adj, order, ring, rs, fused)
		out[n] = [rosettatype(n, info, adj, order, mm), mm, 0.0]
		rt.add_atom(n)
		rt.atom(n).element_type(cm.element_set('default').element(
			info[n]['el']))
		rt.atom(n).formal_charge(info[n]['q'])
	for k in sorted((tuple(sorted(b, key=lambda m: info[m]['i'])), o)
			for b, o in order.items()):
		rt.add_bond(k[0][0], k[0][1], bn[k[1]])
	for n in keep: rt.set_atom_type(n, out[n][0])
	rosetta_recharge_fullatom(rt)
	for n in keep: out[n][2] = rt.atom(n).charge()
	return {n: tuple(v) for n, v in out.items()}


def semirot(chis, info, adj, order):
	'''
	Decide whether the last chi is non rotameric, which is true when it
	rotates a flat sp2 carbon group such as an aromatic ring, a
	carboxylate or an amide, and false for an sp3 or phosphate terminus
	Arguments:
	----------
		chis: list - chi axes from findchi
		info: dict - per atom properties from digest
		adj: dict - atom name to set of bonded atom names
		order: dict - frozenset of two atom names to bond order
	Returns:
	--------
		bool: True when the residue is semi rotameric
	'''
	if not chis: return False
	c = chis[-1][2]
	return info[c]['el'] == 'C' and (info[c]['aro'] or any(
		order.get(frozenset((c, m)), 1) >= 2 for m in adj[c]))


def wells(chis, info, adj, order, ring):
	'''
	Choose the starting rotamer wells and the search grid for each chi
	Arguments:
	----------
		chis: list - chi axes from findchi
		info: dict - per atom properties from digest
		adj: dict - atom name to set of bonded atom names
		order: dict - frozenset of two atom names to bond order
		ring: set - ring bonds as frozensets of two atom names
	Returns:
	--------
		list: one list of well centres in degrees per chi
		list: one low, high, step search grid per chi
	'''
	cen, grid = [], []
	for a, b, c, d in chis:
		# A torsion is in plane if EITHER atom of the rotated bond is a
		# planar centre, so both b and c have to be tested. Testing c
		# alone gave phosphotyrosine's chi3, CE2-CZ-OH-P, staggered
		# wells off the oxygen while missing the aromatic carbon on the
		# other side. Only carbon and nitrogen are planar centres: a
		# phosphate stays tetrahedral despite its P=O, which is why the
		# element test is there.
		flat = lambda x: info[x]['el'] in ('C', 'N') and (info[x]['aro']
			or any(order.get(frozenset((x, m)), 1) >= 2 for m in adj[x]))
		sp2 = flat(b) or flat(c)
		# Two wells for an in-plane chi, not four. Pose's binchi
		# encoding holds three values per chi and maps 0 and 90 onto the
		# same one, so a four well declaration silently collides. 0 and
		# 180 are distinct under it and are the two a flat group has.
		cen.append([0.0, 180.0] if sp2 else [60.0, 180.0, 300.0])
	# 30 degrees up to three chi, 60 beyond. An earlier version refined
	# any residue with an in-plane chi to 30, because the four well
	# declaration then in use put wells at 90 and 270 that a 60 degree
	# grid never visits. In-plane wells are now [0, 180], which a 60
	# degree grid does sample, and the minimiser reaches the +-90 minima
	# from the 60 and 120 starts regardless: phosphotyrosine's chi3 came
	# out sharply bimodal at +-90 on the coarse grid, with 0.63 per cent
	# of its mass trans. The refinement bought nothing and cost a factor
	# of sixteen in starts for a four chi residue, 20736 against 1296.
	step = 30.0 if len(chis) <= 3 else 60.0
	grid = [(0.0, 360.0 - step, step) for _ in chis]
	return cen, grid


def writeopts(path, name, chis, cen, grid, semi, sym, nbb=2):
	'''
	Write the MakeRotLib options file describing the whole scan
	Arguments:
	----------
		path: str - where to write the options file
		name: str - the residue type name to scan
		chis: list - chi axes from findchi
		cen: list - starting well centres per chi
		grid: list - low, high and step of the search grid per chi
		semi: bool - True to treat the last chi as non rotameric
		sym: list - rotational period in degrees of each chi
		nbb: int - number of backbone dihedrals, two for an alpha amino acid
	Returns:
	--------
		Nothing, writes the options file to path
	'''
	n = len(chis)
	out = ['AA_NAME %s' % name, 'NUM_CHI %d' % n, 'NUM_BB %d' % nbb,
		'OMG_RANGE 180 180 1', 'PHI_RANGE -180 170 10',
		'PSI_RANGE -180 170 10', 'EPS_RANGE 180 180 1',
		'TEMPERATURE %g' % KBT]
	if semi: out.append('SEMIROTAMERIC')
	for k in range(n):
		lo, hi, st = grid[k]
		if semi and k == n - 1:
			# A two fold terminal group repeats every 180 degrees, so
			# spending all 36 density bins on the full circle wastes
			# half of them on a copy. Rosetta's own entries for the
			# carboxylates and the aromatic rings are 5 degree bins over
			# 180; the rest are 10 degree bins over 360. Match the
			# period, and nrchi records whichever was used.
			lo, hi, st = ((0.0, 175.0, 5.0) if sym[k] == 180
				else (0.0, 350.0, 10.0))
		out.append('CHI_RANGE %d %g %g %g' % (k + 1, lo, hi, st))
	for k in range(n - 1 if semi else n):
		out.append('ROTWELLS %d %d %s' % (k + 1, len(cen[k]),
			' '.join('%g' % x for x in cen[k])))
	open(path, 'w').write('\n'.join(out) + '\n')


def chirality(xyz):
	'''
	Decide whether the residue is an L or a D amino acid
	Arguments:
	----------
		xyz: dict - atom name to cartesian coordinate
	Returns:
	--------
		str: the Rosetta property, L_AA or D_AA
	'''
	a = [xyz['N'][k] - xyz['CA'][k] for k in range(3)]
	b = [xyz['C'][k] - xyz['CA'][k] for k in range(3)]
	c = [xyz['CB'][k] - xyz['CA'][k] for k in range(3)]
	cr = (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
		a[0] * b[1] - a[1] * b[0])
	# The signed volume of N, C and CB about CA. Every L amino acid in
	# the CCD gives about +2.60 and its mirror image gives -2.60, so the
	# sign alone separates them. Declaring L over D coordinates would be
	# silently wrong in exactly the way a virtual atom is.
	return 'L_AA' if sum(cr[k] * c[k] for k in range(3)) > 0 else 'D_AA'


def params(tri, uni, info, adj, order, xyz, keep, chis, ty):
	'''
	Render a Rosetta residue parameter file for the component
	Arguments:
	----------
		tri: str - the three letter component code
		uni: str - the one letter code to advertise
		info: dict - per atom properties from digest
		adj: dict - atom name to set of bonded atom names
		order: dict - frozenset of two atom names to bond order
		xyz: dict - atom name to ideal xyz tuple
		keep: list - kept atom names in CIF order
		chis: list - chi axes from findchi
		ty: dict - atom name to Rosetta type, CHARMM type and charge
	Returns:
	--------
		str: the complete contents of a .params file
	'''
	bl = sorted(tuple(sorted(b, key=lambda m: info[m]['i']))
		for b in order)
	par, seen, q = {'N': None}, {'N'}, ['N']
	while q:
		u = q.pop(0)
		for v in sorted(adj[u], key=lambda m: (info[m]['el'] == 'H',
				info[m]['i'])):
			if v in seen: continue
			seen.add(v); par[v] = u; q.append(v)
	heavy = [n for n in keep if info[n]['el'] != 'H']
	side = [n for n in heavy if n not in ('N', 'CA', 'C', 'O')]
	hyd = [n for n in keep if info[n]['el'] == 'H']
	out = ['NAME %s_ROTLIB' % tri, 'IO_STRING %s %s' % (tri, uni),
		'TYPE POLYMER', 'AA UNK']
	for n in ['N', 'CA', 'C', 'O'] + side + hyd:
		out.append('ATOM %-4s %-4s %-4s %.4f' % (n, ty[n][0], ty[n][1],
			ty[n][2]))
	for x, y in bl: out.append('BOND %s %s' % (x, y))
	for i, c in enumerate(chis):
		out.append('CHI %d %s %s %s %s' % (i + 1, c[0], c[1], c[2], c[3]))
	out.append('PROPERTIES PROTEIN ALPHA_AA %s' % chirality(xyz))
	out.append('NBR_ATOM CB')
	out.append('NBR_RADIUS %.4f' % (max(math.dist(xyz[n], xyz['CB'])
		for n in keep if n in xyz) + 1.0))
	out.append('FIRST_SIDECHAIN_ATOM CB')
	out.append('LOWER_CONNECT N')
	out.append('UPPER_CONNECT C')
	fmt = 'ICOOR_INTERNAL %5s %11.6f %11.6f %11.6f %5s %5s %5s'
	out.append(fmt % ('N', 0.0, 0.0, 0.0, 'N', 'CA', 'C'))
	out.append(fmt % ('CA', 0.0, 180.0,
		geometry(xyz, 'CA', 'N', 'CA', 'N')[0], 'N', 'CA', 'C'))
	d, th, _ = geometry(xyz, 'C', 'CA', 'N', 'C')
	out.append(fmt % ('C', 0.0, th, d, 'CA', 'N', 'C'))
	out.append(fmt % ('UPPER', 149.999985, 63.800018, 1.328685,
		'C', 'CA', 'N'))
	out.append(fmt % ('LOWER', -150.0, 58.300003, 1.328685,
		'N', 'CA', 'C'))
	# The CCD component is a free amino acid, so its N is an amine and
	# its C a carboxylate: the CIF torsions for the amide H and the
	# carbonyl O describe a zwitterion and say nothing about a peptide
	# bond. Build both off the polymer connections instead, which keeps
	# the peptide unit planar by construction rather than by accident.
	amide = [h for h in hyd if par.get(h) == 'N'][:1]
	out.append(fmt % ('O', -180.0, 59.200005, 1.231015,
		'C', 'CA', 'UPPER'))
	for h in amide:
		out.append(fmt % (h, -180.0, 60.849998, 1.010000,
			'N', 'CA', 'LOWER'))
	skip = set(['N', 'CA', 'C', 'O']) | set(amide)
	for n in [x for x in par if x not in skip]:
		p = par[n]
		if p == 'N': g, gg = 'CA', 'C'
		else:
			g = par[p]
			gg = par[g] or ('C' if g == 'N' else 'N')
		d, th, ph = geometry(xyz, n, p, g, gg)
		out.append(fmt % (n, ph, th, d, p, g, gg))
	return '\n'.join(out) + '\n'



CLASS = {}
for _g, _ns in (('Csp3', 'CT CT1 CT2 CT3 CP1 CP2 CP3'),
	('Caro', 'CA CY CPT CPH1 CPH2'), ('Csp2', 'C CC CD'),
	('Oest', 'ON2 OS'), ('Ohyd', 'OH1'), ('Oanion', 'ON3 OC O OB'), ('P', 'P'),
	('S', 'S SM'), ('N', 'N NH1 NH2 NH3 NC2 NR1 NR2 NR3 NY NP'),
	('Hap', 'HA HB HP HR1 HR3 HC H HS')):
	for _x in _ns.split(): CLASS[_x] = _g


def mmdir():
	'''
	Locate the CHARMM parameter directory inside the Rosetta database
	Arguments:
	----------
		No arguments taken
	Returns:
	--------
		str: path to the fa_standard molecular mechanics parameter set
	'''
	import pyrosetta
	return os.path.join(os.path.dirname(pyrosetta.__file__), 'database',
		'chemical', 'mm_atom_type_sets', 'fa_standard')


def loadtorsions(path):
	'''
	Read a CHARMM torsion parameter file into lookup tables
	Arguments:
	----------
		path: str - the mm_torsion_params.txt to read
	Returns:
	--------
		dict: fully assigned quadruplet to its list of parameter triples
		dict: central pair to its list of parameter triples, for wildcards
	'''
	full, wild = {}, {}
	for l in open(path):
		t = l.split('#')[0].split('!')[0].split()
		if len(t) < 7: continue
		k = tuple(t[:4])
		if k[0] == 'X' and k[3] == 'X':
			wild.setdefault((k[1], k[2]), []).append(t[4:7])
		else: full.setdefault(k, []).append(t[4:7])
	return full, wild


def dihedrals(adj, keep):
	'''
	Every proper torsion inside the residue
	Arguments:
	----------
		adj: dict - atom name to set of bonded atom names
		keep: list - kept atom names
	Returns:
	--------
		set: tuples of four atom names
	'''
	out = set()
	for b in keep:
		for c in adj[b]:
			if c not in keep or c <= b: continue
			for a in adj[b]:
				if a == c or a not in keep: continue
				for d in adj[c]:
					if d != b and d in keep: out.add((a, b, c, d))
	return out


def donor(q, full):
	'''
	Closest shipped quadruplet to take parameters from, matched on the
	chemistry of the rotated bond
	Arguments:
	----------
		q: tuple - the four CHARMM types with no parameters
		full: dict - the fully assigned shipped quadruplets
	Returns:
	--------
		tuple: the donor quadruplet, or None when nothing matches
		list: the donor parameter triples, or None
	'''
	cq = [CLASS.get(x, x) for x in q]
	best, bs, par = None, -1, None
	for k in full:
		# A barrier of this size is not a torsion, it is CHARMM's way of
		# holding an aromatic substituent planar. Copying one onto a
		# rotatable bond would freeze it solid, so such entries are
		# never donors. Leave-one-out over the 924 shipped quadruplets
		# shows this is the whole tail of the error: without the guard
		# the worst barrier recovery is off by 996.9 kcal.
		if max(float(t[0]) for t in full[k]) >= 100.0: continue
		ck = [CLASS.get(x, x) for x in k]
		for cand, cc in ((k, ck), (k[::-1], ck[::-1])):
			if cc[1] != cq[1] or cc[2] != cq[2]: continue
			sc = (2 * (cand[1] == q[1]) + 2 * (cand[2] == q[2])
				+ (cc[0] == cq[0]) + (cc[3] == cq[3])
				+ (cand[0] == q[0]) + (cand[3] == q[3]))
			if sc > bs: best, bs, par = cand, sc, full[k]
	return best, par


def supply(info, adj, keep, ty, work):
	'''
	Find torsions CHARMM cannot parameterise and supply them by analogy
	from the closest shipped entry, so that the scan is not silently
	missing a term
	Arguments:
	----------
		info: dict - per atom properties from digest
		adj: dict - atom name to set of bonded atom names
		keep: list - kept atom names
		ty: dict - atom name to Rosetta type, CHARMM type and charge
		work: str - scratch directory to write the extra parameters into
	Returns:
	--------
		list: one record per supplied torsion, for the output provenance
	'''
	full, wild = loadtorsions(os.path.join(mmdir(), 'mm_torsion_params.txt'))
	miss = set()
	for a, b, c, d in dihedrals(adj, keep):
		q = (ty[a][1], ty[b][1], ty[c][1], ty[d][1])
		if q in full or q[::-1] in full: continue
		if (q[1], q[2]) in wild or (q[2], q[1]) in wild: continue
		miss.add(q)
	if not miss: return []
	lines, rec = [], []
	for q in sorted(miss):
		dn, par = donor(q, full)
		if dn is None:
			sys.exit('[-] Error: no CHARMM torsion for %s and no analogue'
				% '-'.join(q))
		for v in par:
			lines.append('%s %s %s %s %s %s %s'
				% (q[0], q[1], q[2], q[3], v[0], v[1], v[2]))
		rec.append({'torsion': '-'.join(q), 'copied_from': '-'.join(dn),
			'params': [' '.join(v) for v in par]})
	open(os.path.join(work, 'mm_torsion_params.txt'), 'w').write(
		'\n'.join(lines) + '\n')
	log('[+] supplied %d CHARMM torsions by analogy' % len(rec))
	return rec


def binchi(c):
	'''
	Place a chi angle into one of the three rotamer wells, using exactly
	the partition that Pose applies at scoring time
	Arguments:
	----------
		c: float - chi angle in degrees
	Returns:
	--------
		int: 1, 2 or 3, the index of the matching well
	'''
	c = ((c + 180.0) % 360.0) - 180.0
	if 0.0 <= c <= 120.0: return 1
	if abs(c) >= 120.0: return 2
	return 3


def symflag(sym):
	'''
	Render the MakeRotLib symmetry flag for the terminal chi
	Arguments:
	----------
		sym: list - rotational period in degrees of each chi
	Returns:
	--------
		str: the command line fragment, empty when there is no symmetry
	'''
	# Only the last chi may be folded. Rosetta's symmetry options rewrite
	# one chi in place, which is sound for a terminal group but wrong for
	# an interior one: folding phosphotyrosine's chi2 by the ring flip
	# moves CE2, the reference atom of chi3, so the two conformers being
	# merged are not in fact equivalent. Rosetta also honours only one of
	# these options per run and silently discards the second, so emitting
	# both never worked.
	if not sym: return ''
	k, v = len(sym), sym[-1]
	if v == 180:
		return ' -make_rot_lib:two_fold_symmetry_0_180 %d' % k
	if v == 120:
		return ' -make_rot_lib:three_fold_symmetry_90_210_330 %d' % k
	return ''


def relax(work, name):
	'''
	Relax the internal coordinates of a residue against CHARMM equilibria
	Arguments:
	----------
		work: str - directory holding res.params, receives relaxed.json
		name: str - the residue type name declared in that params file
	Returns:
	--------
		Writes work/relaxed.json, mapping atom name to relaxed xyz
	'''
	import pyrosetta
	pf = os.path.join(work, 'res.params')
	pyrosetta.init('-mute all -extra_res_fa %s' % pf, silent=True)
	from pyrosetta.rosetta.core.kinematics import MoveMap
	from pyrosetta.rosetta.core.id import DOF_Type
	from pyrosetta.rosetta.protocols.minimization_packing import MinMover
	from pyrosetta.rosetta.core.scoring import ScoreFunction, ScoreType
	# An idealised CIF block puts every angle at a tetrahedral 109.5, and
	# a single experimental conformer carries one crystal's strain. Both
	# are only seeds. Relaxing against mm_bend and mm_stretch recovers the
	# equilibrium geometry, and is not circular the way cart_bonded would
	# be: those terms read CHARMM equilibria keyed on the mm atom types,
	# not the ICOOR block being repaired. Only bond angles and lengths are
	# freed, so the torsions the scan is about are left untouched.
	pose = pyrosetta.pose_from_sequence('A' + 'X[%s]' % name + 'A')
	sf = ScoreFunction()
	sf.set_weight(ScoreType.mm_bend, 1.0)
	sf.set_weight(ScoreType.mm_stretch, 1.0)
	sf.set_weight(ScoreType.mm_lj_intra_rep, 1.0)
	mm = MoveMap()
	mm.set_bb(False); mm.set_chi(False); mm.set_jump(False)
	mm.set(DOF_Type.PHI, False)
	mm.set(DOF_Type.THETA, True)
	mm.set(DOF_Type.D, True)
	before = sf(pose)
	MinMover(mm, sf, 'lbfgs_armijo_nonmonotone', 1e-6, True).apply(pose)
	res = pose.residue(2)
	out = {res.atom_name(k).strip(): [res.xyz(k).x, res.xyz(k).y,
		res.xyz(k).z] for k in range(1, res.natoms() + 1)}
	json.dump({'xyz': out, 'before': before, 'after': sf(pose)},
		open(os.path.join(work, 'relaxed.json'), 'w'))


def scan(work, name, extra):
	'''
	Run Rosetta's MakeRotLib over the whole backbone grid
	Arguments:
	----------
		work: str - scratch directory holding the params and options files
		name: str - the residue type name to scan
		extra: str - directory of supplementary CHARMM parameters, or ''
	Returns:
	--------
		Nothing, writes one rotlib file per backbone bin into work
	'''
	import pyrosetta
	# MakeRotLib's default pose is one residue carrying the caps as
	# patches, on which every two body term is identically zero:
	# fa_atr, fa_rep, fa_sol and all four hbond terms. The three residue
	# ACE-X-NME form under -make_rot_lib:use_terminal_residues brings
	# them alive and is arguably the better reading of the paper's model
	# system, but measured over six canonicals it costs 8.75 points of
	# masked modal rotamer accuracy and pushes the set below a zero
	# information baseline. It is therefore off, and that is a measured
	# choice rather than an oversight.
	flags = ('-mute all -make_rot_lib:options_file %s/opts.in '
		'-score:weights %s -extra_res_fa %s/res.params -overwrite '
		'-make_rot_lib:output_logging false' % (work, WEIGHTS, work))
	if extra: flags += ' -score:extra_mm_params_dir %s' % extra
	flags += os.environ.get('MRL_SYM', '')
	pyrosetta.init(flags, silent=True)
	from pyrosetta.rosetta.protocols import jd2, make_rot_lib
	jd = jd2.JobDistributor.get_instance()
	jd.set_job_outputter(jd2.NoOutputJobOutputter())
	jd.go(make_rot_lib.MakeRotLibMover(False))


def samples(dens, low, st, k):
	'''
	Split a terminal chi density into a fixed number of samples
	Arguments:
	----------
		dens: list - probability of each of the 36 bins
		low: float - the chi value of bin zero, in degrees
		st: float - the width of one bin, in degrees
		k: int - how many samples to emit
	Returns:
	--------
		list: one tuple of centre, spread and weight per sample
	'''
	# Dunbrack lists a fixed number of terminal chi samples per rotameric
	# well -- six for the two fold aromatics, twelve for the rest -- not a
	# variable set of peaks. Collapsing the density to its modes instead
	# emitted one row where the reference has six, so the library offered
	# a fraction of the conformers a packer expects. Divide the density
	# into k equal arcs and report each one's circular mean and spread.
	m = len(dens)
	w = m // k
	out = []
	for g in range(k):
		sl = dens[g * w:(g + 1) * w]
		tot = sum(sl)
		if tot <= 0.0:
			out.append((low + st * (g * w + (w - 1) / 2.0), st * w, 0.0))
			continue
		sx = sy = 0.0
		for i, p in enumerate(sl):
			a = math.radians(low + st * (g * w + i))
			sx += p * math.cos(a); sy += p * math.sin(a)
		mu = math.degrees(math.atan2(sy, sx))
		r = math.sqrt(sx * sx + sy * sy) / tot
		sd = math.degrees(math.sqrt(max(-2.0 * math.log(max(r, 1e-9)), 0.0)))
		out.append((mu, min(max(sd, SIGMIN), st * w), tot))
	return out


def parserotlib(work, n, semi, low=-180.0, st=10.0, nsamp=12):
	'''
	Read every rotlib file MakeRotLib wrote and index it by backbone bin
	Arguments:
	----------
		work: str - scratch directory holding the rotlib files
		n: int - the number of chi angles
		semi: bool - True when the last chi is non rotameric
		low: float - origin of the terminal chi window, in degrees
		st: float - width of one terminal chi bin, in degrees
		nsamp: int - terminal chi samples to emit per rotameric well
	Returns:
	--------
		dict: bin index to a list of rows, each row holding the well
		      numbers, probability, chi means, sigmas and any density
	'''
	nrot = n - 1 if semi else n
	out = collections.defaultdict(list)
	raw = collections.defaultdict(dict)
	for f in glob.glob(os.path.join(work, '*.rotlib')):
		for line in open(f):
			t = line.split()
			if len(t) < 8 or t[0] != 'UNK': continue
			phi, psi = float(t[1]), float(t[2])
			i = int(round((phi + 180.0) / 10.0)) % 36
			j = int(round((psi + 180.0) / 10.0)) % 36
			p = 4
			if semi:
				rw = [int(x) for x in t[p:p + nrot]]; p += nrot
				prob = float(t[p]); p += 1
				chi = [float(x) for x in t[p:p + nrot]]; p += nrot
				sig = [float(x) for x in t[p:p + nrot]]; p += nrot
				dens = [float(x) for x in t[p:p + 36]]
				# rotate the scanned 0..span grid onto the origin
				# the payload declares, so bin 0 really is chi_last_low
				# index j of the scanned grid holds chi = st*j, and we
				# want it first when st*j == low, so j = low/st mod 36.
				# Negating that is wrong but invisible on the full
				# circle, where 18 and -18 are the same rotation.
				rot = int(round(low / st)) % 36
				dens = dens[rot:] + dens[:rot]
				# File the density under the well the chi actually
				# landed in, not the well the minimisation started
				# from. MakeRotLib labels a row by its starting
				# centroid, and a cluster that begins at trans and
				# slides into gauche minus keeps the trans label: on
				# tryptophan that mis-files 27% of rows carrying 31%
				# of the probability. Pose looks the density up by
				# binchi of the observed chi, so the two must agree.
				key = ','.join(str(binchi(c)) for c in chi)
				# Two clusters can land in the same well of the same
				# backbone bin. merge() handles that on the rotameric
				# branch but never runs here, so a plain assignment let
				# the second silently replace the first: P_rot summed to
				# 0.59 instead of 1, 15% of bins sat flat at the -logP
				# ceiling carrying no information, and the cells that
				# lost their only cluster were zero filled, putting an
				# eclipsed chi of exactly 0.0 into a table fa_dun
				# interpolates over. Combine them instead: probabilities
				# add, chi and sigma average by weight, densities mix.
				b = i * 36 + j
				old = raw[key].get(b)
				if old is None:
					raw[key][b] = (prob, chi, sig, dens)
				else:
					op, oc, og, od = old
					t = op + prob
					if t <= 0.0: t = 1.0
					mc = [math.degrees(math.atan2(
						(op * math.sin(math.radians(a))
							+ prob * math.sin(math.radians(c))) / t,
						(op * math.cos(math.radians(a))
							+ prob * math.cos(math.radians(c))) / t))
						for a, c in zip(oc, chi)]
					# Pool, do not average. Two clusters 20 degrees
					# apart each of width 8 describe a distribution of
					# width about 12.8; averaging returns 8 and throws
					# away the separation between their centres.
					mg = []
					for a, c, ca, cc, mm2 in zip(og, sig, oc, chi, mc):
						da = ((ca - mm2 + 180.0) % 360.0) - 180.0
						dc = ((cc - mm2 + 180.0) % 360.0) - 180.0
						v = (op * (a * a + da * da)
							+ prob * (c * c + dc * dc)) / t
						mg.append(math.sqrt(max(v, 0.0)))
					md = [(op * a + prob * c) / t
						for a, c in zip(od, dens)]
					raw[key][b] = (op + prob, mc, mg, md)
				span = st * 36
				for mu, sd, w in samples(dens, low, st, nsamp):
					# report the mode inside the declared window, so the
					# table's terminal chi and the density beside it are
					# on the same convention. Unfolded, phenylalanine's
					# chi2 spanned the full circle while its reference
					# and its own density spanned 180.
					mu = low + ((mu - low) % span)
					out[i * 36 + j].append((rw, prob * w, chi + [mu],
						sig + [sd], dens))
				continue
			else:
				rw = [int(x) for x in t[p:p + 4]][:nrot]; p += 4
				prob = float(t[p]); p += 1
				chi = [float(x) for x in t[p:p + 4]][:n]; p += 4
				sig = [float(x) for x in t[p:p + 4]][:n]
				dens = None
			out[i * 36 + j].append((rw, prob, chi, sig, dens))
	return out, raw


def merge(rs, n):
	'''
	Combine clusters that minimised into the same rotamer well, because
	Pose stores one row per well index per backbone bin
	Arguments:
	----------
		rs: list - the parsed rotamers of one bin, most probable first
		n: int - the number of chi angles
	Returns:
	--------
		list: one entry per distinct well, probabilities summed
	'''
	out = {}
	for rw, prob, chi, sig, dv in rs:
		k = sum(binchi(chi[i]) * (10 ** (3 - i)) for i in range(n))
		if k in out: out[k][1] += prob
		else: out[k] = [rw, prob, chi, sig, dv]
	return sorted((tuple(v) for v in out.values()), key=lambda r: -r[1])


def nrchi(raw, n, twofold=False, ringed=False):
	'''
	Build the payload that Pose reads for semi rotameric residues, which
	lives under Score Parameters rather than the rotamer library
	Arguments:
	----------
		raw: dict - rotameric well key to bin index to parsed record
		n: int - the number of chi angles
	Returns:
	--------
		dict: the FaDunNrchiDensities entry for this residue
	'''
	nd = n - 1
	maxe = 13.815510557964274
	per = {}
	for k in sorted(raw):
		P = [0.0] * 1296
		nl = [maxe] * 1296
		cm = [0.0] * (nd * 1296)
		cs = [SIGMIN] * (nd * 1296)
		dn = [0.0] * (1296 * 36)
		for cell, (prob, chi, sig, dens) in raw[k].items():
			P[cell] = prob
			nl[cell] = min(maxe, -math.log(max(prob, 1e-6)))
			for i in range(nd):
				cm[i * 1296 + cell] = chi[i]
				cs[i * 1296 + cell] = max(SIGMIN, sig[i])
			for j, v in enumerate(dens): dn[cell * 36 + j] = v
		# A well with no cluster in some bin left chi_means at exactly
		# 0.0, an eclipsed chi that no rotamer occupies, and fa_dun
		# bicubic interpolates that table so each such cell drags its
		# neighbours too. Carry the nearest populated bin's value
		# instead: a real minimum of the same surface a few degrees
		# away beats a value the residue never adopts.
		have = [c for c in range(1296) if P[c] > 0.0]
		if have:
			for c in range(1296):
				if P[c] > 0.0: continue
				ci, cj = c // 36, c % 36
				nb = min(have, key=lambda z:
					(z // 36 - ci) ** 2 + (z % 36 - cj) ** 2)
				for i2 in range(nd):
					cm[i2 * 1296 + c] = cm[i2 * 1296 + nb]
					cs[i2 * 1296 + c] = cs[i2 * 1296 + nb]
				# carry the density too. Leaving it all zero made Pose
				# clamp every bin of it to 1e-6, a flat 13.8 plateau
				# that doubles the penalty neglogP_rot already applies
				# and then rings into live neighbours through the
				# bicubic interpolation.
				dn[c * 36:(c + 1) * 36] = dn[nb * 36:(nb + 1) * 36]
		per[k] = {'P_rot': P, 'neglogP_rot': nl, 'chi_means': cm,
			'chi_sigmas': cs, 'densities': dn}
	# Report the grid that was actually scanned, not a fixed one. A two
	# fold terminal group is sampled at 5 degrees over 180 and lands on
	# a -90 origin after the half array rotation; everything else is 10
	# degrees over 360 and lands on -180. energy.py reads these three
	# numbers out of the payload rather than assuming, so a library is
	# self consistent either way, but a wrong label would misplace every
	# density bin.
	step = 5.0 if twofold else 10.0
	low = (-30.0 if twofold and ringed else
		-90.0 if twofold else -180.0)
	return {'chi_last_low': low, 'chi_last_step': step,
		'chi_last_n': 36, 'n_chi': n, 'n_disc_chi': nd,
		'rotwells': sorted(raw), 'phi_step': 10.0, 'psi_step': 10.0,
		'phi_n': 36, 'psi_n': 36, 'per_rot': per}


def buildtable(rows, n, semi, sym=None):
	'''
	Turn the parsed rotamers into Pose's table and offset arrays
	Arguments:
	----------
		rows: dict - bin index to parsed rotamer rows
		n: int - the number of chi angles
		semi: bool - True when the last chi is non rotameric
	Returns:
	--------
		list: one row per rotamer per bin
		list: 1297 offsets marking where each bin starts
		list: 36 by 36 chi vectors of the most probable rotamer per bin
		list: the 36 bin density rows, empty when rotameric
	'''
	table, offs, top = [], [0], []
	sym = sym or [360] * n
	for i in range(36):
		row = []
		for j in range(36):
			rs = sorted(rows.get(i * 36 + j, []), key=lambda r: -r[1])
			# Canonicalise a two fold interior chi before anything else.
			# Flipping an aromatic ring maps (chi_k, chi_k+1) to
			# (chi_k+180, chi_k+1+180) because the flip carries the
			# reference atom of the next chi, so the pair moves together.
			# This has to happen before merge, so that flip equivalent
			# conformers collapse instead of being listed twice, and
			# before the well index is packed, or the index describes a
			# different conformer from the chi beside it.
			if not semi:
				fr = []
				for rw, prob, chi, sig, dv in rs:
					ch = list(chi)
					for k in range(len(ch) - 1):
						if sym[k] != 180: continue
						v = ((ch[k] + 180.0) % 360.0) - 180.0
						if -90.0 <= v < 90.0: continue
						ch[k] = ((v + 360.0) % 360.0) - 180.0
						ch[k + 1] = ((ch[k + 1] + 360.0) % 360.0) - 180.0
					fr.append((rw, prob, ch, sig, dv))
				rs = merge(fr, n)
			for rw, prob, chi, sig, _ in rs:
				nr = len(rw)
				if semi:
					idx = sum(binchi(chi[k]) * (10 ** (nr - 1 - k))
						for k in range(nr))
				else:
					idx = sum(binchi(chi[k]) * (10 ** (3 - k))
						for k in range(n))
				table.append([idx, round(prob, 6)]
					+ [round(c, 1) for c in chi]
					+ [round(max(s, SIGMIN), 1) for s in sig])
			offs.append(len(table))
			row.append([round(c, 1) for c in rs[0][2]] if rs else [0.0] * n)
		top.append(row)
	return table, offs, top


def emit(tri, n, chis, sym, table, offs, top, semi, sha, added,
		nrc, grid, src, cen, relaxed):
	'''
	Assemble the JSON document that Pose's Parameterise consumes
	Arguments:
	----------
		tri: str - the three letter component code
		n: int - the number of chi angles
		chis: list - chi axes from findchi
		sym: list - rotational period of each chi, provenance only
		table: list - the rotamer rows
		offs: list - the 1297 bin offsets
		top: list - the 36 by 36 most probable chi vectors
		semi: bool - True when the last chi is non rotameric
		sha: str - sha256 of the input CIF
		added: list - any CHARMM torsions supplied for this residue
		nrc: dict - the semirotameric density payload, or None
		grid: tuple - the chi sampling grid the scan used
		src: str - the CIF coordinate block the geometry was seeded from
		cen: list - the starting well centres the scan actually used
		relaxed: bool - True when the geometry relaxation ran
	Returns:
	--------
		dict: the complete document, ready for json.dump
	'''
	out = {'tricode': tri, 'n_chi': n,
		'method': {'pipeline': 'NCAA_PyRosetta.py',
			'protocol': 'Rosetta MakeRotLib (Renfrew 2012), driven via jd2',
			'scorefxn': os.environ.get('MRL_WTS') or 'inline:make_rot_lib_elec',
			'scorefxn_sha256': wtshash(), 'temperature_kT': KBT,
			'semirotameric': semi,
			'chi_axes': [list(c) for c in chis],
			'chi_symmetry_deg': sym,
			'rotwell_encoding': 'sum(binchi(chi_k) * 10**(3-k)); '
				'a semirotameric residue packs 10**(n_rot-1-k) over '
				'its rotameric chi only',
			'columns_note': 'in rotamers.table the column named count '
				'holds the packed well index, not an observation '
				'count; in densities.table it holds the number of '
				'terminal chi samples merged into that row. Pose '
				'fixes both names',
			'phi_grid': [-180, 10, 36], 'psi_grid': [-180, 10, 36],
			'chi_grid': list(grid[0]),
			'chi_wells': cen,
			'kT_note': 'kT 0.60 from the Renfrew 2012 protocol capture '
				'(SI S2 run log); not fitted. Code default is 1.4, '
				'docs say 1. kT rescales populations only: well '
				'centres and rotamer ranking are invariant to it',
			'added_torsions': added, 'cif_sha256': sha,
			'geometry': {'seed': src, 'relaxed': 'mm_bend+mm_stretch'
				'+mm_lj_intra_rep, angles and lengths'
				if relaxed is True else
				'attempted, no CHARMM bond angle; reseeded from the '
				'experimental block' if relaxed == 'failed' else
				'not attempted, beta branched, CIF seed used as is',
				'backbone_icoor': 'H and O built on LOWER/UPPER'}},
		'rotamers': {
			'columns': ['count', 'prob']
				+ ['chi%d' % (k + 1) for k in range(n)]
				+ ['sig%d' % (k + 1) for k in range(n)],
			'table': table, 'bin_offsets': offs, 'top_chi': top},
		'densities': None}
	if semi:
		# One row per backbone bin and rotameric well, carrying only the
		# discrete chis, with its own 36 bin density beside it. The
		# terminal chi is not a rotamer here: it lives entirely in the
		# density. Pose's own entries are strictly one density row per
		# table row in the same order, so density_bins is built here
		# rather than carried over from the rotamer table, which had a
		# different length and a different per bin ordering.
		dt, do, dd = [], [0], []
		nd = n - 1
		for b in range(len(offs) - 1):
			grp = {}
			for r in table[offs[b]:offs[b + 1]]:
				g = grp.setdefault(int(r[0]),
					[0, 0.0] + [0.0] * 3 * nd)
				g[0] += 1; g[1] += r[1]
				for k in range(nd):
					# accumulate chi on the unit circle: a cluster
					# straddling +-180 averaged linearly lands near
					# zero, which put one tryptophan well at -26.9
					# degrees instead of -180
					a = math.radians(r[2 + k])
					g[2 + 2 * k] += r[1] * math.cos(a)
					g[3 + 2 * k] += r[1] * math.sin(a)
					g[2 + 2 * nd + k] += r[1] * r[2 + n + k]
			for key in sorted(grp):
				c, pr = grp[key][0], grp[key][1]
				w = pr if pr else 1.0
				# split the packed well key back into one column per
				# rotameric chi, which is the shape Pose stores
				ks = [int(x) for x in str(key).zfill(nd)]
				dt.append(ks + [c, round(pr, 6)]
					+ [round(math.degrees(math.atan2(
						grp[key][3 + 2 * k], grp[key][2 + 2 * k])), 1)
						for k in range(nd)]
					+ [round(max(grp[key][2 + 2 * nd + k] / w,
						SIGMIN), 1) for k in range(nd)])
				# per_rot is keyed comma joined, the same way
				# parserotlib built it and energy.py splits it. The
				# packed decimal index is a different encoding: str(12)
				# never matches '1,2', so a two chi lookup missed every
				# time and the fallback below wrote a flat 1/36 over
				# real data. One rotameric chi hides it, because
				# str(1) == '1'.
				pv = (nrc or {}).get('per_rot', {}).get(
					','.join(str(x) for x in ks), {})
				dv = (pv.get('densities') or [])[b * 36:(b + 1) * 36]
				if len(dv) != 36:
					sys.exit('[-] Error: no density for well %s bin %d'
						% (ks, b))
				dd.append([round(x, 6) for x in dv])
			do.append(len(dt))
		# Pose's density_grids are [-180,10,36], [-90,5,36], [-30,5,36].
		# Emitting 0 while the payload declares a -90 or -30 origin is
		# self contradictory metadata.
		gid = {-180.0: 0, -90.0: 1, -30.0: 2}.get(
			nrc['chi_last_low'] if nrc else -180.0, 0)
		out['densities'] = {'grid_id': gid,
			'columns': ['r%d' % (k + 1) for k in range(nd)]
				+ ['count', 'prob']
				+ ['chi%d' % (k + 1) for k in range(nd)]
				+ ['sig%d' % (k + 1) for k in range(nd)],
			'table': dt, 'bin_offsets': do, 'density_bins': dd}
		out['FaDunNrchiDensities'] = nrc
	return out


def rosetta_build(tri, cif, out_path=None):
	'''
	Build a rotamer library for one component with Rosetta MakeRotLib
	Arguments:
	----------
		tri: three letter component code, e.g. PTR
		cif: path to that component's wwPDB CCD CIF
		out_path: file to write, or None for stdout
	Returns:
	--------
		Nothing, writes a JSON document to out_path or stdout
	'''
	tri = tri.upper()
	if not os.path.exists(cif): sys.exit('[-] Error: no such file %s' % cif)
	sha = hashlib.sha256(open(cif, 'rb').read()).hexdigest()
	import pyrosetta
	pyrosetta.init('-mute all', silent=True)
	atoms, bonds = readcif(cif, tri)
	if not atoms: sys.exit('[-] Error: %s holds no component %s' % (cif, tri))
	info, adj, order, xyz, keep, src = digest(atoms, bonds)
	miss = [a for a in BACKBONE + ('CB',) if a not in keep]
	if miss: sys.exit('[-] Error: %s lacks %s' % (tri, ', '.join(miss)))
	keep, dropped = protonate(info, adj, order, keep, xyz)
	ring = bridges(adj, keep, info)
	chis = findchi(info, adj, order, ring)
	n = len(chis)
	if not 1 <= n <= 4:
		sys.exit('[-] Error: %s has %d chi, MakeRotLib supports 1 to 4'
			% (tri, n))
	sym = findsym(chis, info, adj, order, ring, keep)
	ty = types(info, adj, order, keep, ring)
	semi = semirot(chis, info, adj, order)
	log('[+] %s: %d atoms, %d chi %s%s' % (tri, len(keep), n,
		['-'.join(c) for c in chis],
		', semirotameric' if semi else ''))
	if dropped: log('[+] ionised: dropped %s' % ', '.join(dropped))
	relaxed = False
	# Fail here rather than deep inside Rosetta with an unrelated
	# message. WEIGHTS resolves next to the script, so moving the script
	# without its weight file lands exactly here.
	if os.environ.get('MRL_WTS') and not os.path.exists(WEIGHTS):
		sys.exit('[-] Error: weight file %s not found' % WEIGHTS)
	nb = len([a for a in adj['CB']
		if a != 'CA' and info[a]['el'] != 'H'])
	dorelax = RELAX is True or (RELAX == 'auto' and nb < 2)
	if RELAX == 'auto':
		log('[+] geometry: CB carries %d heavy substituents, %s'
			% (nb, 'relaxing' if dorelax else 'keeping CIF seed'))
	work = tempfile.mkdtemp(prefix='mrl_')
	open(os.path.join(work, 'res.params'), 'w').write(
		params(tri, 'X', info, adj, order, xyz, keep, chis, ty))
	if dorelax:
		rr = subprocess.run([sys.executable, os.path.abspath(__file__),
			'--relax', work, '%s_ROTLIB' % tri], cwd=work,
			capture_output=True, text=True)
		rj = os.path.join(work, 'relaxed.json')
		# mm_bend needs a CHARMM bond angle for every triple, and the
		# residues with missing torsions tend to be missing angles too.
		# That is not a reason to abandon the build: fall back to the
		# unrelaxed seed, say so loudly, and record it in the output so
		# the geometry provenance is never silently wrong.
		if not os.path.exists(rj):
			sys.stderr.write(rr.stdout[-1200:] + rr.stderr[-1200:])
			# The idealised block is only preferred because relaxation
			# repairs it downstream. With no relaxation that defence is
			# gone, so fall back to the experimental coordinates rather
			# than keep the seed we cannot fix. For phosphotyrosine the
			# difference is the whole ball game: CZ-OH-P is 106.8 in the
			# idealised block against 125.9 experimentally, and the PDB
			# mean over 128 structures is 125.9.
			info, adj, order, xyz, keep, src = digest(
				atoms, bonds, 'model_Cartn_x')
			keep, dropped = protonate(info, adj, order, keep, xyz)
			log('[!] no relaxation for %s; reseeding from %s' % (tri, src))
			relaxed = 'failed'
			open(os.path.join(work, 'res.params'), 'w').write(
				params(tri, 'X', info, adj, order, xyz, keep, chis, ty))
		else:
			rd = json.load(open(rj))
			xyz = {k: tuple(v) for k, v in rd['xyz'].items() if k in xyz}
			log('[+] geometry relaxed: mm_bend+mm_stretch %.3f -> %.3f'
				% (rd['before'], rd['after']))
			relaxed = True
			open(os.path.join(work, 'res.params'), 'w').write(
				params(tri, 'X', info, adj, order, xyz, keep, chis, ty))
	added = supply(info, adj, keep, ty, work)
	cen, grid = wells(chis, info, adj, order, ring)
	writeopts(os.path.join(work, 'opts.in'), '%s_ROTLIB' % tri, chis,
		cen, grid, semi, sym)
	log('[+] scanning 1296 backbone bins with MakeRotLib ...')
	env = dict(os.environ)
	if added: env['MMEXTRA'] = work
	# Honour an inherited value so the symmetry flag can be overridden
	# for a sweep, the way MRL_WTS can. Assigning unconditionally made
	# the documented override a no-op.
	env['MRL_SYM'] = os.environ.get('MRL_SYM') or symflag(sym)
	if env['MRL_SYM']: log('[+] symmetry:%s' % env['MRL_SYM'])
	r = subprocess.run([sys.executable, os.path.abspath(__file__),
		'--scan', work, '%s_ROTLIB' % tri], cwd=work, env=env,
		capture_output=True, text=True)
	# MakeRotLib also writes one <NAME>_definitions.rotlib alongside the
	# 1296 bins, so a plain count of 1296 would pass with a bin missing.
	got = [g for g in glob.glob(os.path.join(work, '*.rotlib'))
		if not g.endswith('_definitions.rotlib')]
	if len(got) != 1296:
		sys.stderr.write(r.stdout[-2000:] + r.stderr[-2000:])
		sys.exit('[-] Error: MakeRotLib produced %d of 1296 bins'
			% len(got))
	two = semi and sym[-1] == 180
	# Pose puts an aromatic ring's terminal chi on a -30 origin and a
	# carboxylate's on -90, both over 180 degrees. Ring membership of
	# the pivot atom is what separates them.
	ringed = semi and any(frozenset((chis[-1][2], m)) in ring
		for m in adj[chis[-1][2]])
	tlo = -30.0 if (two and ringed) else -90.0 if two else -180.0
	tst = 5.0 if two else 10.0
	rows, raw = parserotlib(work, n, semi, tlo, tst,
		6 if two else 12)
	table, offs, top = buildtable(rows, n, semi, sym)
	doc = emit(tri, n, chis, sym, table, offs, top, semi, sha,
		added, nrchi(raw, n, sym[-1] == 180, ringed) if semi else None, grid,
		src, cen,
		relaxed)
	if out_path:
		with open(out_path, 'w') as fh:
			json.dump(doc, fh, separators=(',', ':'))
	else:
		json.dump(doc, sys.stdout, separators=(',', ':'))
		sys.stdout.write('\n')
	log('[+] done: %d rows over %d bins' % (len(table), len(offs) - 1))
	shutil.rmtree(work, ignore_errors=True)




def pipeline_rosetta(cif, tricode, out_path, log):
	"""
	Drive Rosetta MakeRotLib for one residue
	Arguments:
	----------
		cif: path to the component CIF
		tricode: three letter component code
		out_path: file to write
		log: logger
	Returns:
	--------
		Nothing, writes a JSON document to out_path
	"""
	try:
		import pyrosetta          # noqa: F401
	except ImportError as e:
		sys.exit('[-] Error: --rosetta needs PyRosetta (%s).\n'
			'    Install it with:  bash setup.sh' % e)
	rosetta_build(tricode, cif, out_path)
	log.info('wrote %s' % out_path)



# ----------------------------------------------------------------------
# SwissSidechain (--swiss)
# ----------------------------------------------------------------------

SWISS_URL = ('https://www.swisssidechain.ch/data/download/'
	'L_bbdep_Gfeller.lib.zip')
# Axes for the three residues SwissSidechain is normally used for. Any
# other code falls back to axes derived from the CIF bond graph, which
# is the same detection --rosetta uses.
SWISS_AXES = {
	'ORN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
		['CB', 'CG', 'CD', 'NE']],
	'PTR': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1'],
		['CE2', 'CZ', 'OH', 'P'], ['CZ', 'OH', 'P', 'O1P']],
	'TPO': [['N', 'CA', 'CB', 'OG1'], ['CA', 'CB', 'OG1', 'P'],
		['CB', 'OG1', 'P', 'O1P']]}


def swiss_axes(tri, cif):
	"""
	Chi axes for a SwissSidechain residue, preferring the known set
	Arguments:
	----------
		tri: three letter component code
		cif: path to that component's CIF, used only as a fallback
	Returns:
	--------
		list: chi axes, each a list of four atom names
	"""
	if tri in SWISS_AXES:
		return SWISS_AXES[tri]
	atoms, bonds = readcif(cif, tri)
	if not atoms:
		sys.exit('[-] Error: %s holds no component %s' % (cif, tri))
	info, adj, order = digest(atoms, bonds)
	return [list(c) for c in findchi(info, adj, order, findrings(info, adj))]


def pipeline_swiss(cif, tricode, out_path, log):
	"""
	Download SwissSidechain and convert one residue to the Pose schema
	Arguments:
	----------
		cif: path to the component CIF, for chi axis fallback
		tricode: three letter component code
		out_path: file to write
		log: logger
	Returns:
	--------
		Nothing, writes a JSON document to out_path
	"""
	import io, urllib.request, zipfile
	axes = swiss_axes(tricode, cif)
	n = len(axes)
	log.info('downloading %s' % SWISS_URL)
	archive = urllib.request.urlopen(SWISS_URL, timeout=900).read()
	lib = zipfile.ZipFile(io.BytesIO(archive)).read('L_bbdep_Gfeller.lib')
	log.info('  %.0f MB uncompressed' % (len(lib) / 1e6))
	# The published grid is 37 x 37 because phi and psi both run -180 to
	# +180 inclusive and -180 is the same cell as +180. Drop the -180
	# edge and wrap +180 onto index 0 to reach Pose's 36 x 36.
	cells = [[] for _ in range(1296)]
	for line in lib.decode('utf-8', 'replace').splitlines():
		f = line.split()
		if not f or f[0] != tricode:
			continue
		phi, psi = float(f[1]), float(f[2])
		if phi == -180.0 or psi == -180.0:
			continue
		i = int(round((phi + 180) / 10)) % 36
		j = int(round((psi + 180) / 10)) % 36
		row = [int(f[3]), float(f[8])]
		row += [float(f[9 + k]) for k in range(n)]
		row += [float(f[13 + k]) for k in range(n)]
		cells[i * 36 + j].append(row)
	got = sum(len(c) for c in cells)
	if not got:
		sys.exit('[-] Error: SwissSidechain holds no residue %s' % tricode)
	log.info('  extracted %d rotamer lines for %s' % (got, tricode))
	empty = [b for b in range(1296) if not cells[b]]
	if empty:
		sys.exit('[-] Error: %d of 1296 bins are empty for %s'
			% (len(empty), tricode))
	table, offsets, top_chi = [], [0], []
	for i in range(36):
		row_top = []
		for j in range(36):
			cell = sorted(cells[i * 36 + j], key=lambda r: -r[1])
			total = sum(r[1] for r in cell) or 1.0
			for r in cell:
				table.append([r[0], round(r[1] / total, 9)]
					+ [round(v, 4) for v in r[2:]])
			offsets.append(len(table))
			row_top.append([round(v, 4) for v in cell[0][2:2 + n]])
		top_chi.append(row_top)
	doc = {
		'tricode': tricode,
		'n_chi': n,
		'densities': None,
		'rotamers': {
			'columns': ['count', 'prob']
				+ ['chi%d' % (k + 1) for k in range(n)]
				+ ['sig%d' % (k + 1) for k in range(n)],
			'table': table,
			'bin_offsets': offsets,
			'top_chi': top_chi},
		'method': {
			'pipeline': 'ncaarotamers.py --swiss',
			'source': 'SwissSidechain backbone-dependent rotamer library',
			'url': SWISS_URL,
			'chi_axes': axes,
			'grid': '36 x 36 at 10 deg, indexed phi * 36 + psi',
			'column0': 'observation count per (phi, psi) cell',
			'phi_grid': [-180, 10, 36],
			'psi_grid': [-180, 10, 36]}}
	with open(out_path, 'w') as fh:
		json.dump(doc, fh, separators=(',', ':'))
	log.info('%s : %d rotamers over %d bins, %d chi'
		% (out_path, len(table), len(offsets) - 1, n))



# ----------------------------------------------------------------------
# Main entry-point
# ----------------------------------------------------------------------

def main():
	"""
	Parse arguments and dispatch to the selected pipeline
	Arguments:
	----------
		No arguments taken
	Returns:
	--------
		Nothing, writes a rotamer library to <TRICODE>.json
	"""
	# MakeRotLib is driven by re-invoking this file as a subprocess, so
	# these two internal modes are matched before argparse ever runs.
	# They are not part of the public interface.
	if len(sys.argv) == 4 and sys.argv[1] == '--relax':
		return relax(sys.argv[2], sys.argv[3])
	if len(sys.argv) == 4 and sys.argv[1] == '--scan':
		return scan(sys.argv[2], sys.argv[3],
			os.environ.get('MMEXTRA', ''))
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
	pipe.add_argument('--rosetta', action='store_true',
		help='Rosetta MakeRotLib (Renfrew 2012), needs PyRosetta')
	pipe.add_argument('--swiss', action='store_true',
		help='Download and convert the SwissSidechain library')
	args = ap.parse_args()
	log = setup_logging('ncaarotamers')
	tricode = args.tricode.upper()
	if not os.path.exists(args.cif):
		sys.exit('[-] Error: no such file %s' % args.cif)
	if (args.dft or args.md or args.denovo) and not HAVE_NN:
		sys.exit('[-] Error: --dft, --md and --denovo need numpy, gemmi, '
			'rdkit, ase and torchani (%s).\n'
			'    Install them with:  bash setup.sh' % NN_ERR)
	out_path = '%s.json' % tricode
	if args.dft:
		pipeline_dft(args.cif, tricode, out_path, log)
	elif args.md:
		pipeline_md(args.cif, tricode, out_path, log)
	elif args.rosetta:
		pipeline_rosetta(args.cif, tricode, out_path, log)
	elif args.swiss:
		pipeline_swiss(args.cif, tricode, out_path, log)
	else:
		pipeline_denovo(args.cif, tricode, out_path, log)


if __name__ == '__main__':
	main()

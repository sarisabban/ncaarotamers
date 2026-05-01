# ncaarotamers

One script. Three pipelines. Backbone-dependent rotamer libraries for non-canonical amino acids from a single CIF file.

```bash
python ncaarotamers.py --cif cifs/ALY.cif --tricode ALY --denovo
```

| Flag | Hardware | Wall-clock | Accuracy |
|---|---|---|---|
| `--dft` | National HPC (≈1024 cores + 32–64 GPUs) | 1–3 weeks | DFT gold standard |
| `--md` | 1–4 datacenter GPUs (A100 / H100) | 1–3 days | ≥90 % of `--dft` |
| `--denovo` | Single laptop (8 CPUs, no GPU required) | minutes–hours | well centres ≈ 5–10° vs DFT-on-tripeptide |

## Quick start

```bash
git clone https://github.com/sarisabban/ncaarotamers
cd ncaarotamers
bash setup.sh                              # one-time: venv + all deps
source venv/bin/activate
python ncaarotamers.py --cif cifs/VAL.cif --tricode VAL --denovo
```

## When to use which pipeline

* **You have a national HPC allocation**: `--dft`. RESP charges, ωB97X-D / aug-cc-pVTZ DFT, TIP4P-Ew explicit MD, full free-energy decomposition. Reference / gold standard.
* **You have a workstation with 1–4 GPUs and a few days**: `--md`. Same protocol as `--dft` but with a neural-network potential (ANI-2x) replacing every DFT call. ≈100–1000× cheaper for ≥90 % of the accuracy on organic chemistry.
* **You only have a laptop**: `--denovo`. NN-potential constrained chi scan, gas-phase, no MD. Well centres are correct; populations are gas-phase Boltzmann and may diverge from protein-context Dunbrack on solvation or packing-coupled chi axes.

## Examples

Same CLI for every pipeline; only the flag changes.

```bash
# Tier 1 (HPC)
python ncaarotamers.py --cif cifs/ALY.cif --tricode ALY --dft

# Tier 2 (GPU workstation)
python ncaarotamers.py --cif cifs/ALY.cif --tricode ALY --md

# Tier 3 (laptop)
python ncaarotamers.py --cif cifs/ALY.cif --tricode ALY --denovo
```

For your own residue, drop the CIF in `cifs/` and run:

```bash
python ncaarotamers.py --cif cifs/MY_RESIDUE.cif --tricode XYZ --denovo
```

## How it works

All three pipelines start from the same physical model: an Ace-X-Nme tripeptide built from the input CIF, with explicit hydrogens and chemically-sensible cap geometry. They then differ in what energy method drives the (φ, ψ, χ) conformational scan.

### `--dft` (Tier 1, HPC)

Reference / gold standard. Steps:

1. Parse the CIF via `gemmi`; build the residue in `RDKit`; attach ACE / NME caps to construct an Ace-X-Nme tripeptide.
2. Derive RESP charges from an HF/6-31G(d) PCM ESP fit via [Psi4](https://psicode.org/) and the OpenFF Recharge tools. Combine with [OpenFF Sage 2.x](https://openforcefield.org/) for bonded parameters.
3. **FF↔QM gate**: cross-validate the FF against ωB97X-D / aug-cc-pVTZ DFT single-points on 200 random conformations. Refuse to proceed if RMSE > 1 kcal/mol or correlation < 0.95.
4. **DFT relaxed scan** at every (φ, ψ) bin in the 36 × 36 grid: enumerate canonical chi seeds, run two-stage Psi4 optimisation (constrained then released), single-point energy at ωB97X-D / aug-cc-pVTZ, analytical Hessian for chi sigmas, cluster minima by L∞ distance.
5. **MD validation** in TIP4P-Ew explicit water at every bin: restrain (φ, ψ), 100 ns × 3 replicates, histogram chi values, compute MD-derived populations.
6. **Free energy** per well: `A = E_DFT + ZPE + S_vib(RR-HO) + ΔG_solv(MD basin)`. Boltzmann populations at 300 K.
7. Emit `rot_v1` JSON.

Compute footprint: ≈1024 cores + 32–64 GPUs, 1–3 weeks. The script the other two are measured against.

### `--md` (Tier 2, GPU workstation)

Reproduces the `--dft` structural protocol step-for-step, replacing every DFT call with a neural-network-potential call:

| Step | --dft | --md |
|---|---|---|
| Charges | RESP from Psi4 ESP | not needed (NN-pot has no per-residue parameters) |
| Optimisation | ωB97X-D / ma-def2-TZVP, PCM | ANI-2x via ASE L-BFGS |
| Single-point E | ωB97X-D / aug-cc-pVTZ, PCM | ANI-2x single-point |
| Hessian | analytical (Psi4) | numerical (ASE finite-difference) |
| MD | TIP4P-Ew, OpenFF, 100 ns × 3 reps | TIP3P (or gas-phase fallback), [openmm-ml](https://github.com/openmm/openmm-ml) `MLPotential('ani2x')`, 100 ps × 1 rep |
| Free energy | A = E + ZPE + S_vib + ΔG_solv | same |

Why this works: ANI-2x (Devereux et al., JCTC 2020) is a neural-network potential trained on ~9 million ωB97X-level DFT calculations for organic chemistry. It returns DFT-quality energies and forces in ~5 ms per call vs ~minutes for real DFT — ≥90 % of the accuracy at ~0.1 % of the compute cost.

Compute envelope: ≈9–12 hours on 4× A100 for the full pipeline at 36 × 36 grid with 100 ps MD per top well.

#### v1 limitations of `--md`

* OpenMM's `Modeller.addSolvent` cannot template arbitrary small-molecule LIG residues, so explicit-water solvation can fall back to gas-phase MD. The MD still runs and produces thermal sampling; the dielectric screening is just absent. Charged sidechains may show populations off by ~1 kcal/mol per chi until the openff-interchange wiring is added in a future release.
* Hessian-based sigmas are computed at one representative bin (alpha-helix, φ = -60°, ψ = -45°) and broadcast to the entire grid rather than per bin. Per-bin Hessians would 30× the compute cost; for most NCAAs the chi sigmas are weakly (φ, ψ)-dependent so this is an acceptable trade-off.

### `--denovo` (Tier 3, laptop)

NN-potential constrained chi scan, gas-phase, no MD. The cheapest useful method:

1. Parse CIF, build Ace-X-Nme tripeptide.
2. Auto-detect chi axes from the side-chain bond graph.
3. At every (φ, ψ) bin: restrain φ/ψ via harmonic potential, restrain chi axes at canonical seeds (g+, t, g-), L-BFGS minimise using the NN-potential as the energy function, then release the chi restraints (keeping a weak retention term to preserve rotamer-well topology) and re-minimise to the local well.
4. Cluster minima by L∞ distance, Boltzmann-weight populations, emit JSON.

Chi well **centres** within ~5–10° of DFT-on-tripeptide. What you don't get: protein-context populations. Gas-phase tripeptide minima can diverge from folded-protein observations on bins where solvation or adjacent-residue packing biases the wells.

## Output schema

`output/<TRICODE>.json`, rot_v1 schema, byte-compatible with Pose's
`database.json["Rotamer Library"]["residues"][tricode]`:

```jsonc
{
    "tricode":  "VAL",
    "n_chi":    1,
    "rotamers": {
        "columns":     ["r1", "count", "prob", "chi1", "sig1"],
        "table":       [/* CSR-packed: bin_idx = i_phi*36 + i_psi */],
        "bin_offsets": [/* 1297 ints: 36*36 + 1 */],
        "top_chi":     [/* 36 x 36 x n_chi */]
    },
    "densities": null,
    "method":    {/* tier, citations, parameters used */},
    "metadata":  {/* phi/psi grid, sigma floor */}
}
```

To insert into Pose:

```python
from Pose.pose.tools import Parameterise
Parameterise(
    cif_file='cifs/VAL.cif',
    rotamer_json_file='output/VAL.json',
    tricode='VAL',
    unicode='V',
)
```

## Dependencies

Single `requirements.txt`. `setup.sh` creates a venv and installs everything in one command:

```bash
bash setup.sh
```

If `psi4` fails to install (the wheel is sometimes platform-specific), either use a conda environment for the `--dft` pipeline:

```bash
conda create -n ncaarotamers -c conda-forge psi4 openff-toolkit openff-recharge openff-units mdtraj
pip install -r requirements.txt
```

…or skip Tier 1 entirely (Tier 2 / Tier 3 don't need psi4 / openff*):

```bash
grep -v -E '^psi4|^openff|^mdtraj' requirements.txt > req.tmp
python3 -m venv venv
source venv/bin/activate
pip install -r req.tmp
```

The script auto-detects what's installed: `--denovo` and `--md` work
without psi4; `--dft` raises a clear error message if the DFT stack
is missing.

## Repository layout

```
ncaarotamers/
├── README.md          this file
├── LICENSE            MIT
├── requirements.txt   all deps for all three pipelines
├── setup.sh           one-command venv setup
├── .gitignore
├── ncaarotamers.py    THE script
├── cifs/              example CIFs from RCSB CCD
│   ├── ALY.cif
│   ├── LYS.cif
│   └── VAL.cif
└── output/            JSON outputs land here (gitignored)
```

## License

MIT. See [LICENSE](LICENSE).

## References

### Methodology

* Shapovalov MV, Dunbrack RL Jr. **A smoothed backbone-dependent rotamer library for proteins derived from adaptive kernel density estimates and regressions.** *Structure* 19:844–858 (2011). [doi:10.1016/j.str.2011.03.019](https://doi.org/10.1016/j.str.2011.03.019)
* Renfrew PD, Choi EJ, Bonneau R. **Incorporation of noncanonical amino acids into Rosetta and use in computational protein-peptide interface design.** *PLoS ONE* 7:e32637 (2012). [doi:10.1371/journal.pone.0032637](https://doi.org/10.1371/journal.pone.0032637)
* Bayly CI, Cieplak P, Cornell W, Kollman PA. **A well-behaved electrostatic potential based method using charge restraints for deriving atomic charges: the RESP model.** *J. Phys. Chem.* 97:10269–10280 (1993).

### DFT and force fields (`--dft`)

* Mardirossian N, Head-Gordon M. **ωB97X-V: a 10-parameter, range-separated hybrid, generalized gradient approximation density functional with nonlocal correlation, designed by a survival-of-the-fittest strategy.** *Phys. Chem. Chem. Phys.* 16:9904–9924 (2014).
* Smith DGA, Burns LA, Simmonett AC, et al. **Psi4 1.4: open-source software for high-throughput quantum chemistry.** *J. Chem. Phys.* 152:184108 (2020). [psicode.org](https://psicode.org/)
* Wagner J, Thompson MW, Dotson DL, et al. **OpenFF: a flexible scientific framework for molecular force field development.** *J. Chem. Theory Comput.* (2024). [openforcefield.org](https://openforcefield.org/)
* Marenich AV, Cramer CJ, Truhlar DG. **Universal solvation model based on solute electron density and on a continuum model of the solvent.** *J. Phys. Chem. B* 113:6378–6396 (2009).

### Neural-network potentials (`--md` / `--denovo`)

* Devereux C, Smith JS, Davis KK, et al. **Extending the applicability of the ANI deep learning molecular potential to sulfur and halogens.** *J. Chem. Theory Comput.* 16:4192–4202 (2020). [doi:10.1021/acs.jctc.0c00121](https://doi.org/10.1021/acs.jctc.0c00121)
* Smith JS, Isayev O, Roitberg AE. **ANI-1: an extensible neural network potential with DFT accuracy at force field computational cost.** *Chem. Sci.* 8:3192–3203 (2017).
* Anstine D, Zubatyuk R, Isayev O. **AIMNet2: a neural network potential for charged, hyper-valent and unusual chemical environments.** *Nat. Comm.* (2024).

### Software stacks

* Larsen AH, Mortensen JJ, Blomqvist J, et al. **The Atomic Simulation Environment — a Python library for working with atoms.** *J. Phys. Condens. Matter* 29:273002 (2017). [wiki.fysik.dtu.dk/ase](https://wiki.fysik.dtu.dk/ase/)
* Eastman P, Galvelis R, Peláez RP, et al. **OpenMM 8: molecular dynamics simulation with machine learning potentials.** *J. Chem. Theory Comput.* (2024). [openmm.org](https://openmm.org/)
* Galvelis R, Varela-Rial A, Doerr S, et al. **NNP/MM: accelerating molecular dynamics simulations with machine learning potentials and molecular mechanics.** *J. Chem. Theory Comput.* (2023). The openmm-ml integration. [github.com/openmm/openmm-ml](https://github.com/openmm/openmm-ml)
* Landrum G et al. **RDKit: open-source cheminformatics.** [rdkit.org](https://www.rdkit.org/)
* Wojdyr M. **Gemmi: a library for crystallographic and molecular models.** [gemmi.readthedocs.io](https://gemmi.readthedocs.io/)

### Data

* Burley SK, Bhikadiya C, Bi C, et al. **RCSB Protein Data Bank.** *Nucleic Acids Res.* 51:D488 (2023). CC0 1.0. [rcsb.org](https://www.rcsb.org/)

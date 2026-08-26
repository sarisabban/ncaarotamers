# ncaarotamers

Backbone-dependent rotamer libraries for amino acids, canonical or not, from a single CIF file. Output is JSON in the format [Pose](https://github.com/sarisabban/Pose) reads.

## Install

```bash
"${SHELL}" <(curl -L micro.mamba.pm)
bash setup.sh
mamba activate ncaarot
```

This builds one environment with every dependency pinned in `requirements.yml`.

## Run

Get the CIF for your residue from the wwPDB, then pick one method:

```bash
wget https://files.rcsb.org/ligands/download/PTR.cif

python ncaarotamers.py --cif PTR.cif --tricode PTR --denovo
```

| Method | What it uses | Hardware | Time |
|---|---|---|---|
| `--denovo` | neural-network potential, gas phase | laptop | minutes–hours |
| `--md` | neural-network potential + MD | 1–4 GPUs | 1–3 days |
| `--dft` | DFT + explicit-solvent MD | HPC cluster | 1–3 weeks |
| `--rosetta` | Rosetta MakeRotLib | laptop | 1 chi ≈ 1 min, 4 chi ≈ hours |
| `--swiss` | downloads the SwissSidechain library | laptop | minutes |

Results are written to `./<TRICODE>.json`.

## Output

Every method writes the same five top-level keys: `['tricode', 'n_chi', 'rotamers', 'densities', 'method']`. `rotamers` holds `columns`, `table`, `bin_offsets` and `top_chi`, on a 36 x 36 grid of backbone angles indexed `phi * 36 + psi`.

## Notes

`--rosetta` does not use the score function distributed with the MakeRotLib protocol. The distributed one carries only half of a matched CHARMM term, which skews chi1. The replacement improves SER, PHE and TYR but makes HIS and TRP worse. Measured against the canonical reference library, 8 of 17 residues beat a backbone-blind control on all chi angles, and 13 of 17 on chi1 alone. Treat chi1 as validated and chi2 onward as unvalidated. All of that testing is on canonical residues; nothing here shows it carries over to non-canonical ones.
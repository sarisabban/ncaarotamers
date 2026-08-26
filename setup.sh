#!/usr/bin/env bash
# One-command environment for ncaarotamers.
#
#   ./setup.sh                  create the env, install everything
#   ./setup.sh -n myenv         use a different env name
#   ./setup.sh --no-pyrosetta   skip PyRosetta
#   ./setup.sh --relock         regenerate requirements.yml from the
#                               spec inlined in this file
#
# Conda-forge, not pip. psi4, openff-toolkit and openff-recharge are
# not on PyPI, so a venv cannot install this dependency set at all;
# requirements.txt used to promise exactly that and could not deliver
# it. Only PyRosetta comes from pip, because it is not on conda-forge
# either.
#
# PyRosetta is NOT open-source. It is distributed under the Rosetta
# Software Non-Commercial License Agreement: free for not-for-profit
# research institutions, government laboratories and universities, and
# for individuals not acting on behalf of a for-profit entity.
# Commercial use needs a separate licence from UW CoMotion
# (license@uw.edu). Installing it is your acceptance of that
# agreement. Set PYROSETTA_ACCEPT_LICENSE=1 to confirm without the
# prompt, or pass --no-pyrosetta to skip it.
#
# Needs micromamba, mamba or conda on PATH. To get micromamba:
#   "${SHELL}" <(curl -L micro.mamba.pm)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

ENVNAME=ncaarot
WANT_PYROSETTA=1
RELOCK=0
while [ $# -gt 0 ]; do
	case "$1" in
		-n|--name) ENVNAME="$2"; shift 2 ;;
		--no-pyrosetta) WANT_PYROSETTA=0; shift ;;
		--relock) RELOCK=1; shift ;;
		-h|--help) sed -n '2,24p' "$0"; exit 0 ;;
		*) echo "unknown option: $1" >&2; exit 1 ;;
	esac
done

CONDA=""
for c in micromamba mamba conda; do
	if command -v "$c" >/dev/null 2>&1; then CONDA="$c"; break; fi
done
if [ -z "$CONDA" ]; then
	echo "need micromamba, mamba or conda on PATH." >&2
	echo 'install micromamba:  "${SHELL}" <(curl -L micro.mamba.pm)' >&2
	exit 1
fi
echo "using $CONDA"

# Refuse to write into an env that already exists. Creating over the
# environment that built the published libraries is not recoverable.
if "$CONDA" env list | awk '{print $1}' | grep -qx "$ENVNAME"; then
	echo "env '$ENVNAME' already exists." >&2
	echo "choose another with -n NAME, or remove it:" >&2
	echo "  $CONDA env remove -n $ENVNAME" >&2
	exit 1
fi

# Everything comes from requirements.yml, a conda-lock file pinning all
# ~330 packages, transitive dependencies and build hashes included, for
# linux-64, osx-arm64 and osx-64. Naming versions in this script instead
# would pin only the two dozen named here and let the other three
# hundred float; that is how a pytorch pin of 2.10.0 silently became
# 2.13.0 during testing.
#
# The spec below is the input to that lock. It lives here rather than in
# a second checked-in file so the repo carries one dependency file, not
# two. Edit it, then run ./setup.sh --relock to regenerate
# requirements.yml.
SPEC=$(cat <<'YML'
name: ncaarot
channels:
  - conda-forge
dependencies:
  # Pinned to the versions the published libraries were built with.
  - python=3.11.15
  - numpy=2.4.6
  - scipy=1.17.1
  - rdkit=2026.3.1
  - openmm=8.5.2
  - openff-toolkit=0.18.0
  - openff-interchange=0.5.2
  - openff-units=0.3.2
  - openff-nagl=0.5.5
  - openff-nagl-models=2025.9.0
  - vina=1.2.7
  - tqdm=4.70.0
  - pytorch=2.10.0
  # Needed by ncaarotamers.py. These were absent from the environment
  # that built the libraries, so the versions below are simply what
  # resolved when the lock was generated, not verified choices.
  - gemmi=0.7.5
  - scikit-learn=1.9.0
  - joblib=1.5.3
  - requests=2.34.2
  - psi4=1.11
  - torchani=2.8.2
  - ase=3.29.0
  - mdtraj=1.11.1
  - openff-recharge=0.5.3
  - pip
  - pip:
      - openmmml @ git+https://github.com/openmm/openmm-ml.git
YML
)
LOCK="$HERE/requirements.yml"

if [ "$RELOCK" -eq 1 ]; then
	TMPSPEC="$(mktemp -d)/environment.yml"
	printf '%s\n' "$SPEC" > "$TMPSPEC"
	if command -v conda-lock >/dev/null 2>&1; then
		CL="conda-lock"
	else
		B="_relock_boot"
		"$CONDA" create -y -n "$B" -c conda-forge python=3.11 pip >/dev/null
		"$CONDA" run -n "$B" python -m pip install --quiet conda-lock==4.0.2
		CL="$CONDA run -n $B conda-lock"
	fi
	$CL lock -f "$TMPSPEC" -p linux-64 -p osx-arm64 -p osx-64 \
		--lockfile "$LOCK"
	[ -n "${B:-}" ] && "$CONDA" env remove -y -n "$B" >/dev/null 2>&1
	echo "regenerated $LOCK"
	exit 0
fi
if [ ! -f "$LOCK" ]; then
	echo "missing $LOCK" >&2
	echo "regenerate it with conda-lock, see the comment in this file" >&2
	exit 1
fi

# The lock must be installed by conda-lock itself. Passing it to
# "mamba create -f" looks like it works and silently creates an EMPTY
# environment, because mamba does not parse conda-lock's multi-platform
# YAML. If conda-lock is not on PATH, bootstrap it into a throwaway env
# rather than requiring the user to install it first.
BOOT=""
if command -v conda-lock >/dev/null 2>&1; then
	CONDA_LOCK="conda-lock"
else
	BOOT="${ENVNAME}_condalock_boot"
	echo "conda-lock not found; bootstrapping into '$BOOT'"
	"$CONDA" create -y -n "$BOOT" -c conda-forge python=3.11 pip >/dev/null
	"$CONDA" run -n "$BOOT" python -m pip install --quiet conda-lock==4.0.2
	CONDA_LOCK="$CONDA run -n $BOOT conda-lock"
fi

$CONDA_LOCK install -n "$ENVNAME" "$LOCK"

if [ -n "$BOOT" ]; then
	"$CONDA" env remove -y -n "$BOOT" >/dev/null 2>&1 || true
fi

# An empty env means the lock was not applied. Fail loudly rather than
# reporting success over a broken environment.
if ! "$CONDA" run -n "$ENVNAME" python -c "import numpy" >/dev/null 2>&1; then
	echo "install produced an environment without numpy; lock not applied" >&2
	exit 1
fi

# Resolve the env's python without needing shell hooks.
PY="$("$CONDA" run -n "$ENVNAME" python -c 'import sys; print(sys.executable)')"

if [ "$WANT_PYROSETTA" -eq 1 ] && [ "${PYROSETTA_ACCEPT_LICENSE:-0}" != "1" ]; then
	echo
	echo "PyRosetta is under the Rosetta Software Non-Commercial License"
	echo "Agreement. Free for academic, government and not-for-profit use;"
	echo "commercial use requires a licence from UW CoMotion."
	read -r -p "Do you qualify and accept? [y/N] " reply
	case "$reply" in
		[yY]|[yY][eE][sS]) ;;
		*) echo "Skipping PyRosetta."; WANT_PYROSETTA=0 ;;
	esac
fi

if [ "$WANT_PYROSETTA" -eq 1 ]; then
	# Not on conda-forge, and not on PyPI either: the installer fetches
	# it from the RosettaCommons mirror. Reference build was
	# 2026.32+release.fa5ce20989.
	"$PY" -m pip install pyrosetta-installer==0.1.2
	"$PY" -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()"
	"$PY" -c "import pyrosetta; print('PyRosetta OK')"
fi

echo
echo "env '$ENVNAME' ready."
echo "Activate with:  $CONDA activate $ENVNAME"
echo "Then run:       python ncaarotamers.py --cif cifs/VAL.cif --tricode VAL --denovo"
if [ "$WANT_PYROSETTA" -eq 1 ]; then
	echo "        or:     python NCAA_PyRosetta.py PTR PTR.cif > ptr_rot.json"
else
	echo
	echo "NCAA_PyRosetta.py needs PyRosetta; rerun without --no-pyrosetta."
fi

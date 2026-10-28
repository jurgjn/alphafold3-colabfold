# Scripts

Standalone analysis scripts for checking predictions and conversions by hand.
They are for developers; none of them runs in CI, and none is needed to run a
model.

| path | purpose |
|---|---|
| [`chirality_check.py`](chirality_check.py) | checks a prediction's stereocentres against the CCD ideal conformer for the named residue codes, which an RMSD gate cannot see |
| [`dl_chirality_score.py`](dl_chirality_score.py) | scores the 5KX0 cyclic D/L peptide: how many L and D residues were built with the right chirality, CA RMSD to native, pTM and mean pLDDT |
| [`of3_verification/`](of3_verification/README.md) | the original OpenFold3 conversion audit and module comparisons (stale: see its README) |

## Usage

```bash
# stereochemistry of one or more predictions
PYTHONPATH=src python scripts/chirality_check.py --codes BTN prediction.cif
PYTHONPATH=src python scripts/chirality_check.py --codes A,G,C,U rna.cif

# D-residue chirality across runs of examples/regression/5kx0_cyclic_dl_peptide.json,
# one run_alphafold.py --output_dir per tag, at $DL_RUNS/dl_<tag>/
DL_RUNS=/path/to/runs python scripts/dl_chirality_score.py <tag> [<tag> ...]
```

**Requirements.** Both need `biopython` and an importable `alphafold3` (for the
CCD). `chirality_check.py` takes it from `PYTHONPATH`; `dl_chirality_score.py`
finds this checkout's `src/` itself, and reads the native structure from
`$AF3_DEV_HOME/5KX0.cif` (default `~/5KX0.cif`; see `dev/af3_dev_paths.py`).

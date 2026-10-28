# Example inputs

Fold-input JSON files, one job each, in the `alphafold3` dialect (version 4)
described in [docs/input.md](../docs/input.md). Each one exercises a different
kind of input, so together they double as a smoke test: `run_alphafold_test.py`
folds all thirteen top-level examples (`test_examples`). Every file sets
`modelSeeds: [42]`, and every top-level one gives each chain a `description`.

## Top-level examples

| File | Job name | What it demonstrates |
|---|---|---|
| [`ubiquitin_monomer.json`](ubiquitin_monomer.json) | `ubiquitin_monomer` | The simplest input: one protein chain (human ubiquitin, 76 aa). |
| [`tetr_homodimer.json`](tetr_homodimer.json) | `TetR_homodimer` | A homodimer written as one entry with a list of chain IDs (`"id": ["A", "B"]`). |
| [`barnase_barstar.json`](barnase_barstar.json) | `barnase_barstar_complex` | A heterodimer: two different protein chains (barnase and barstar). |
| [`erk2_phosphorylated.json`](erk2_phosphorylated.json) | `ERK2_dual_phosphorylated` | Protein post-translational modifications (`ptmType` `TPO` at 185, `PTR` at 187). |
| [`calmodulin_4calcium.json`](calmodulin_4calcium.json) | `calmodulin_4calcium` | Ions as ligands: four copies of CCD `CA` from one ligand entry. |
| [`tetr_dimer_tetracycline.json`](tetr_dimer_tetracycline.json) | `TetR_dimer_tetracycline` | A homodimer with a CCD ligand (`TAC`), one copy per monomer. |
| [`streptavidin_biotin_smiles.json`](streptavidin_biotin_smiles.json) | `streptavidin_biotin_smiles` | A ligand given as a SMILES string (biotin) rather than a CCD code. |
| [`kras_g12c_sotorasib.json`](kras_g12c_sotorasib.json) | `KRAS_G12C_sotorasib_covalent` | A covalent ligand: CCD `MOV` (sotorasib) bonded to Cys12 `SG` through `bondedAtomPairs`. |
| [`rnaseb_glycosylated.json`](rnaseb_glycosylated.json) | `RNaseB_glycosylated` | A glycan: a multi-CCD ligand (`NAG NAG BMA MAN MAN`) bonded to Asn34 `ND2`, with the four bonds inside the glycan. |
| [`u1a_rna_hairpin.json`](u1a_rna_hairpin.json) | `U1A_RNA_hairpin` | Protein-RNA: the U1A RRM1 domain with an RNA stem-loop. |
| [`modified_rna.json`](modified_rna.json) | `modified_RNA_hairpin` | RNA modifications (`PSU`, `5MC`, `OMG`). |
| [`methylated_dna.json`](methylated_dna.json) | `methylated_CpG_dsDNA` | Double-stranded DNA with modifications (`5CM` at the CpG sites of both strands). |
| [`tetr_dimer_dna.json`](tetr_dimer_dna.json) | `TetR_dimer_tetO_DNA` | Protein-DNA: a TetR homodimer on two complementary operator strands. |

None of these carry an MSA or templates, so they need either the local data
pipeline (the genetic databases in
[docs/installation.md](../docs/installation.md)) or the ColabFold MSA server.

## Regression inputs

| File | Job name | What it demonstrates |
|---|---|---|
| [`regression/5kx0_cyclic_dl_peptide.json`](regression/5kx0_cyclic_dl_peptide.json) | `5kx0` | A de novo cyclic peptide (PDB 5KX0) with eleven D-amino acids given as modifications (`DAS`, `DPR`, `DGL`, ...) and an intra-chain disulfide (Cys8 to D-Cys21) in `bondedAtomPairs`. Its MSA and templates are set empty, so it runs without the data pipeline. It is the target where a model's learned L-prior is wrong eleven times; `scripts/dl_chirality_score.py` scores the result (see [docs/ported_models.md](../docs/ported_models.md)). |

## Checking and running an example

Check that every example parses (no GPU or databases needed):

```sh
JAX_PLATFORMS=cpu python -c '
import sys
from alphafold3.common import folding_input
for p in sys.argv[1:]:
  print(p, folding_input.Input.from_json(open(p).read(), json_path=p).name)
' examples/*.json examples/regression/*.json
```

Fold one with MSAs from the ColabFold server instead of local databases (any
`--model`, see the [README](../README.md#the-models-this-fork-runs)):

```sh
python run_alphafold.py \
  --model=boltz2 \
  --json_path=examples/ubiquitin_monomer.json \
  --output_dir=./output \
  --use_msa_server \
  --norun_data_pipeline
```

The regression input needs no MSA at all. `--cyclic=A` wraps chain A's
relative-position encoding (see the README's "Cyclic chains"):

```sh
python run_alphafold.py \
  --model=rosettafold3 \
  --json_path=examples/regression/5kx0_cyclic_dl_peptide.json \
  --output_dir=./output \
  --norun_data_pipeline \
  --cyclic=A
```

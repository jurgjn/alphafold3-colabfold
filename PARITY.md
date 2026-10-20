# Parity

Every model this fork runs is checked against its **own vendor's stock code**, not just for plausible structures. This file holds the current results. The dated notebook of how each finding was made is in [docs/parity_log.md](docs/parity_log.md), and open module-level items are in [dev/oracles/HOLES.md](dev/oracles/HOLES.md).

**Status (2026-10-20):** every model matches its stock vendor on every input that vendor can take. That covers the trunk, one denoise step, confidence, padded inputs and the full diffusion trajectory. There is one caveat, protenix1's trajectory, explained [below](#full-trajectory).

- [How it is measured](#how-it-is-measured)
- [Results by model](#results-by-model)
- [Full trajectory](#full-trajectory)
- [Inputs a vendor cannot take (N/A)](#inputs-a-vendor-cannot-take-na)
- [Vendor bugs the harness patches](#vendor-bugs-the-harness-patches)
- [Vendor conventions the port reproduces](#vendor-conventions-the-port-reproduces)
- [BioEmu2](#bioemu2)
- [AlphaFold 2](#alphafold-2)
- [Module-level gates (L0 to L7)](#module-level-gates-l0-to-l7)
- [Fold accuracy](#fold-accuracy)
- [Re-running](#re-running)

## How it is measured

Two harnesses compare the whole model against stock code:

- `dev/oracles/vendor_native_parity.py` covers every ported model.
- `dev/oracles/af3_native_parity.py` covers `alphafold3` against DeepMind's own repository.

Both sides run their **shipped entry points** on the same input JSON. The vendor side runs in the vendor's own environment, with its own featuriser, model loader and forward pass. Our side runs our featurisation and `Model.__call__`. The harness hands neither side anything except **random draws**:

- the reference-conformer draw;
- the samples that the confidence head scores;
- one noised input for the denoise step.

No flag, constructor argument or feature is chosen by the harness. Atoms are matched by (token, atom name), so an atom that only one side has is reported, never silently dropped.

**Rows graded per case:**

- `s_inputs`;
- the trunk single and pair after recycling;
- distogram logits;
- one denoise step;
- pLDDT, PAE and PDE.

Each row is graded on correlation **and** `max|d|/rms`:

| grade | corr | max\|d\|/rms |
|---|---|---|
| PARITY | ≥ 0.99999 | ≤ 1e-3 |
| CLOSE | ≥ 0.9999 | ≤ 1e-2 |
| LOOSE | ≥ 0.999 | ≤ 1e-1 |
| BAD | otherwise | |

A case PASSes when every row is PARITY or CLOSE. `af3_native_parity.py` is stricter and requires PARITY.

**The 23 cases** come from `af3_native_parity.cases()`. They are deliberately small, so that each one fits a float32 fold on a 23 GB card:

| group | cases |
|---|---|
| chains | `protein` (6MRR), `homodimer`, `heterodimer`, `long_chain` |
| MSA | `msa`, `deep_msa` (more than 1024 rows, so subsampling and crops apply), `paired_msa` (paired across a heterodimer, with species keys) |
| nucleic acids | `rna`, `rna_mod` (PSU), `dna`, `dna_mod` (5CM), `protein_dna` |
| ligands | `ligand_ccd` (ATP), `ligand_smiles`, `ion` (MG), `multi_ligand` |
| modified / unknown | `ptm` (SEP), `unknown` (X) |
| bonds | `covalent` (NAG on Asn), `glycan` (NAG–NAG, two declared bonds) |
| templates | `template` (5K9P, carries SEP), `template_std` (the same, standard residues only), `multi_template` (two hits, one a sub-range) |
| cyclic | `cyclic`: only for vendors whose input can express it |

**Two more checks run on top of the 23 cases:**

- **Padding** (`OURS_BUCKET=shipped`) pads our side exactly as `run_alphafold.py` does. Real tokens must not move.
- **Full trajectory** (`--trajectory`) runs each side's complete sampler at its default settings. Every random draw is injected by meaning: the initial positions, plus each step's augmentation rotation, translation and noise. A different step count, schedule, noise scale or augmentation convention then shows up as a different final structure.

## Results by model

The vendor versions are the ones installed for the harness. All runs used float32 with `JAX_DEFAULT_MATMUL_PRECISION=highest`, on an A10 (24 GB) and an A100 (40 GB).

| `--model` | stock reference | 23 cases | padded | trajectory |
|---|---|---|---|---|
| `alphafold3` | DeepMind alphafold3 v3.0.4-49 | 23/23 (PARITY-only rule) | — | — |
| `boltz2` | boltz 2.2.1 | 23/23, plus `cyclic` | PASS | CLOSE, 0.025 Å |
| `chai1` | chai_lab 0.6.1 | 20/20, 3 N/A | PASS | PARITY, 0.001 Å |
| `protenix2` | Protenix v2.0.0 | 23/23 | PASS | CLOSE, 0.012 Å |
| `protenix1` | Protenix v2.0.0 (v1 checkpoint) | 23/23 | PASS | LOOSE; superposed 0.007 Å, [see below](#full-trajectory) |
| `opendde` | OpenDDE v1.1.0 | 23/23 | PASS | PARITY, 0.000 Å |
| `rosettafold3` | foundry aad357b | 21/21, 2 N/A | PASS | CLOSE, 0.007 Å |
| `openfold3` | OpenFold3 0.4.5 (preview-2) | 21/21, 2 N/A | PASS | PARITY, 0.000 Å |
| `openbind0` | OpenFold3 v0.5.0 | 21/21, 2 N/A | PASS | PARITY, 0.000 Å |
| `intellifold2` | IntelliFold v2.0.4 (native PyTorch) | 23/23 | PASS | PARITY, 0.005 Å |
| `esmfold2` | esm 3.4.1 | 23/23 | PASS | PARITY, 0.000 Å |
| `esmfold2_fast` | esm 3.4.1 | 23/23 | — | PARITY, 0.000 Å |
| `esmfold2_lm600m` | Synthyra FastPLMs | 20/20, 3 N/A | — | PARITY, 0.000 Å |
| `esmfold2_lm300m` | Synthyra FastPLMs | 20/20, 3 N/A | — | PARITY, 0.000 Å |
| `bioemu2` | BioEmu2 @7846769 | 20/20, 3 N/A | PASS | PARITY, 0.000 Å ([modes](#bioemu2)) |
| `af2_ptm`, `af2_multimer` | DeepMind alphafold (AF2) | [18 gates, all green](#alphafold-2) | | |

The padded runs cover `protein`, `ligand_ccd` and `heterodimer`.

## Full trajectory

The trajectory distance is the maximum per-atom distance between the final coordinates. "Superposed" is the RMS after removing the rigid-body difference.

| model | steps | max \|d\| | superposed rms | rigid rotation | grade |
|---|---|---|---|---|---|
| `boltz2` | 200 | 0.025 Å | 0.0001 Å | 0.003° | CLOSE |
| `chai1` | 199 | 0.001 Å | 0.0000 Å | 0.000° | PARITY |
| `protenix2` | 200 | 0.012 Å | 0.0008 Å | 0.047° | CLOSE |
| `protenix1` | 200 | 0.221 Å | 0.0066 Å | 0.908° | LOOSE |
| `opendde` | 200 | 0.000 Å | 0.0000 Å | 0.000° | PARITY |
| `rosettafold3` | 49 | 0.007 Å | 0.0006 Å | 0.005° | CLOSE |
| `openfold3` | 200 | 0.000 Å | 0.0000 Å | 0.000° | PARITY |
| `openbind0` | 200 | 0.000 Å | 0.0000 Å | 0.000° | PARITY |
| `intellifold2` | 201 | 0.005 Å | 0.0002 Å | 0.019° | PARITY |
| `esmfold2` | 10 | 0.000 Å | 0.0000 Å | 0.000° | PARITY |
| `esmfold2_fast` | 10 | 0.000 Å | 0.0000 Å | 0.000° | PARITY |
| `esmfold2_lm600m` | 11 | 0.000 Å | 0.0000 Å | 0.000° | PARITY |
| `esmfold2_lm300m` | 11 | 0.000 Å | 0.0000 Å | 0.000° | PARITY |
| `bioemu2` | 200 | 0.000 Å | 0.0000 Å | 0.000° | PARITY |

**protenix1 is the one LOOSE row.** Its difference is almost entirely a 0.9° rigid rotation of the whole structure; after superposition the remaining difference is 0.0066 Å. To tell model sensitivity from a port bug, the harness nudges the vendor's *own* trunk outputs by 1e-5 (`TRAJ_FLOOR=trunk`):

- protenix1 then rotates itself by **0.38°** (superposed 0.0027 Å);
- protenix2 rotates itself by only 0.025°.

So protenix1's sampler amplifies trunk differences at the 1e-5 level into a global rotation. The single-step rows for the same input are all PARITY or CLOSE. The raw grade is left as LOOSE rather than relaxed.

The step counts are each vendor's own default. Each schedule ends the way its vendor's does (`schedule_tail` in `diffusion_head.py`):

- protenix and opendde replace the last level with 0;
- intellifold2 appends a 0 to N+1 points;
- the ESMFold2 family spaces N points and appends a 0;
- chai evaluates at midpoints.

## Inputs a vendor cannot take (N/A)

A case is N/A only when the **stock** vendor has no way to express the input. Its runner exits with code 3, and that is reported as N/A, never as PASS. A crash is always a FAIL. Our port runs every one of these inputs; there is simply nothing to compare against.

| model | N/A cases | why |
|---|---|---|
| `chai1` | `ligand_ccd`, `ion`, `multi_ligand` | chai takes ligands as SMILES only (`ligand_smiles` PASSes) |
| `openfold3`, `openbind0` | `covalent`, `glycan` | stock OpenFold3 never reads `Query.covalent_bonds` |
| `rosettafold3` | `multi_template` | RF3 templates come from its input structure, and only one templated chain maps |
| `rosettafold3` | `glycan` | the RF3 JSON takes one `ccd_code` per chain |
| `esmfold2_lm600m`, `esmfold2_lm300m` | `msa`, `deep_msa`, `paired_msa` | trained without MSA conditioning; the vendor rejects explicit MSAs |
| `intellifold2` | `cyclic` | IntelliFold parses `cyclic: true`, but its `RelativePositionEncoding` never reads `cyclic_period` |
| `bioemu2` | `template`, `template_std`, `multi_template` | BioEmu2 has no template input |
| all but `boltz2` | `cyclic` | no cyclic input at all |

`--cyclic` is a feature of this fork: every model here honours it, but only boltz2 was trained with it.

## Vendor bugs the harness patches

Stock OpenFold3 0.4.5 has three MSA bugs that would make a correct port read as a failure. All three are fixed in OpenFold3 v0.5.0. 0.4.5 is still the reference because it is the newest release that loads the preview-2 checkpoint. The runner (`openfold3_upstream_runner.py`) patches each bug behind its own flag, and the sweeps set all three. Our port follows the fixed behaviour.

| flag | stock behaviour |
|---|---|
| `OF3_DIAG_FIX_PROFILE` | `calculate_profile` repeats column indices against a row-major ravel, so `profile[c]` counts the wrong cells. It is exact only for a one-row MSA. Fixed upstream in v0.5.0 (f896db32). |
| `OF3_DIAG_FIX_DELETION` | Operator precedence scales the deletion value by 8/π where AF3 (and v0.5.0) uses 2/π: four times too large. |
| `OF3_DIAG_FIX_PAIRED` | `create_paired_from_precomputed` never calls `set_row_counts`, so a precomputed paired MSA is silently dropped. Fixed in v0.5.0. |

Some vendors draw random MSA rows. `MSA_ALL_ROWS=1`, the harness default, lifts each vendor's draw so that both sides see the same rows:

- OF3 `subsample_all_msa`;
- RF3 `n_msa`;
- protenix `sample_indices`;
- OpenDDE `msa_depth` (1280 rows);
- IntelliFold `msa_depth` (1024 rows).

## Vendor conventions the port reproduces

These are the places where a vendor deliberately differs from AlphaFold 3, and where the sweep found that the port had not yet followed it. Each fix is in the model's spec or policy, not in the harness.

- **Templates:** `pipeline/template_policy.py` has one policy per model (`TEMPLATE_POLICIES` in `model_registry`):
  - RF3 takes the first altloc.
  - chai drops any hit that has a modified residue.
  - OF3 and OpenBind read a modified residue as UNK, realign the whole chain, use the parent's one-letter code, and rank hits.
- **OpenFold3's paired MSA:**
  - Rows are stacked as the query block, then the paired block as given (whose row 0 is the query again), then the main MSA minus the paired rows. The query therefore sits in rows 0 and 1.
  - 0.4.5 does not dedupe the main MSA.
- **RF3:**
  - Rows are paired by TaxID and flagged with `residue_is_paired`.
  - Triangle multiplication divides by the *real* token count, not the padded one.
  - Trailing insertions in an a3m are ignored, as AF3's parser does.
- **chai:**
  - The MSA query row is marked paired.
  - Its pair mask comes from the reference mask, so a reference-only leaving atom (NAG O1) is still an attended key.
  - Its rows are ordered by numpy 1.26's unstable quicksort, which is ported exactly.
- **IntelliFold-2:** uses the boltz-fork MSA build, with a per-chain profile over `msa_valid_depths`.
- **protenix:** the seed applies MC dropout on the recycled pair. Replaying it needs torch's fused-dropout vector width: 4 for float32, 8 for bfloat16.
- **Padded atoms:** every non-AF3 model masks padded atom queries.

## BioEmu2

BioEmu2 shares no network stage with AF3, so its harness branch compares its own feature dict and outputs: `bioemu2_case`, `bioemu2_trajectory` and `bioemu2_steer` in `vendor_native_parity.py`, run against `dev/oracles/bioemu2_upstream_runner.py`. Stock BioEmu2 runs in float32 at `highest` matmul precision; its default is a bf16 trunk with a TF32 denoiser.

**Features.** Every array of stock BioEmu2's feature dict equals ours on all 23 inputs, and on 28 of 30 further edge-case inputs (PTMs at chain ends, unknown nucleotides, MSA caps, partial pairing, glycans split across entries, deuterium, tricky SMILES). The other two use HEM, which RCSB revised after AF3's CCD snapshot; stock fetches its components live. That covers CCD-ideal reference positions, OXT and OP3 dropped at every polymer position, atom tokens for ligands, modified residues and glycans, and Boltz-1 MSA pairing with `deletion_value = π/2·atan(d/3)`.

**Network.** Every case reads PARITY on `s_inputs`, the trunk single and pair, the distogram, the pair conditioning and one denoise step, at max\|d\|/rms ≤ 1.1e-4. The padded check (`OURS_BUCKET=shipped`) and contact conditioning (residue–residue, residue–ligand atom) pass too.

**Its other modes**, final coordinates against stock on the same draws:

| run | result |
|---|---|
| structure mode, protein | PARITY, max \|d\| 0.000 Å |
| structure mode, ATP ligand | PARITY, 0.001 Å |
| equilibrium mode, heterodimer | PARITY, 0.001 Å |
| propagation, 1 ns from a stock equilibrium sample | PARITY, 0.000 Å |
| `edm_gd`, 4 particles | PARITY / CLOSE, ≤ 0.009 Å (superposed ≤ 0.0009 Å) |
| `edm_fk` (with FK resampling), 4 particles | PARITY, ≤ 0.001 Å |
| `edm_fkc` (gradient through the denoiser), 4 particles | PARITY, 0.000 Å |
| `edm_fk` / `edm_fkc` under a custom `--potentials` YAML | PARITY, ≤ 0.001 Å |
| `edm_fk` steering a propagation | PARITY, ≤ 0.001 Å |
| structure mode, homodimer | PARITY, 0.000 Å |
| `edm_gd` / `edm_fkc`, homodimer | PARITY / CLOSE, ≤ 0.022 Å |
| `edm_fk`, homodimer | LOOSE: a near-tie resample (below) |

The ten steering potentials match stock directly as well. Energies agree to ≤ 1e-7 relative and gradients to ≤ 4.4e-5, at random coordinates and at a real ligand geometry. Every topology table is equal.

**Steering a ligand is chaotic.** On the ATP case `edm_gd` reads LOOSE: 0.005–0.05 Å superposed against stock. `STEER_FLOOR` nudges stock's own x0 by 1e-5, and stock then moves 0.004–0.03 Å against itself. The flat-bottom restraints have step-function gradients, so a float-level difference that switches one restraint on or off changes a whole guidance step. This is the same class as protenix1's trajectory, and the raw grade is left as it is.

**FK resampling on the homodimer meets a near-tie.** A trace of every resampling step (`BIOEMU2_FK_TRACE`) shows ours and stock drawing identical indices for 45 resamples. At step 135 the weights are ours [0, 0.216, 0.706, 0.078] against stock's [0, 0.207, 0.723, 0.070], and the systematic draw falls between 0.207 and 0.216, so one side keeps particle 1 and the other particle 2. The two lineages end 2.8 Å apart. Stock does the same to itself: with its x0 nudged by 1e-4 Å (`STEER_FLOOR_EPS`) it switches lineage, and in that run ours matched stock to 0.003 Å. Which side of the tie a run lands on is float noise. Each multi-chain potential alone (`SymmetricChainCOM`, `InterChainVDW`, `IntraChainVDW`) reads PARITY, and so does the default set without its four ligand potentials. Those potentials are empty here, but dropping them reorders the energy sum, which is enough to move the tie.

**Physicality report.** Every stat, pass and reason equals stock's `check_physicality`. The check covered six structures (protein, homodimer, ligand, PTM, glycan, protein–DNA), each under 5–7 noise levels, in all three presets.

Two inputs come from outside, and the harness hands over the vendor's copy rather than grading them:

- A SMILES ligand's conformer is an unseeded RDKit draw in stock BioEmu2.
- The ligand restraint bounds are RDKit's. Releases differ (2025.09 here, 2026.03 in the vendor environment); under the vendor's release, ours are exact.

Not ported, and not reachable from stock's own entry points either:

- `ContactPotential` and `TemplateReferencePotential`: no `prepare()` fills their tables.
- Context-parallel sampling.

## AlphaFold 2

None of the AF3 module levels apply to AF2, a sibling network with no converter, so AF2 is graded separately. **L5af2 is 18 gates, all green.**

Thirteen of them compare against DeepMind's own repository:

- the trunk end to end;
- the structure module as eight fold iterations;
- extra-MSA and IPA;
- all five heads;
- all three template paths;
- the MSA pipeline.

They cover both the monomer and multimer paths. That oracle found three real divergences:

- a masked template distogram;
- BERT masking that was implemented but never called;
- an AF3 confidence pairformer that never received the dropout flag.

There are two deliberate differences:

- MSA sizes are 512/1024 on both paths, where stock uses 512/5120 and 508/2048.
- The monomer's TensorFlow row selection is replaced by multimer's gumbel argsort, because there is no TensorFlow here.

## Module-level gates (L0 to L7)

These came before the whole-model harness. They compare one module at a time on bounded inputs, and they localise a failure that the whole-model rows only detect. Last full run: 2026-09-14 (`bash dev/oracles/run_all_parity.sh all`, graded by `dev/oracles/parity_audit.py`).

| level | what it compares |
|---|---|
| L0 | conversion coverage: every checkpoint tensor accounted for, in both directions |
| L1 | the trunk: z-init, pairformer, MSA module, template embedder, distogram |
| L1r | the trunk on a real input: the input embedder and recycling |
| L1x | the same on a complex, where the cross-chain terms come into play |
| L2 | diffusion conditioning, token transformer, atom encoder and decoder |
| L3 | one full denoise step |
| L4 | the confidence head: PAE, PDE, pLDDT, resolved |
| L5 | an end-to-end fold, scored against an experimental structure |
| L6 | modality: RNA, DNA, ligands, complexes, modified residues |
| L7 | the output side: fold, write the mmCIF, parse it back, compare |
| Lg | differentiability in the sequence (the design path): 16/16 models |

```
369 cells    295 OK   1 SKIP   73 N/A   0 FAIL
346 comparisons: PARITY=304  CLOSE=14  FLOOR=17  DIAG=6  LOOSE=5  BAD=0
```

How to read these labels:

- **N/A** means the model has no such module. `dev/oracles/gate_applies.py` decides this from the registry, so an empty cell can never quietly mean "not run".
- **FLOOR** means the cell has no resolution left: nudging the gate's own input by 1e-6 moves its output further than the port differs. All 17 FLOOR cells are the trunk pairformer on synthetic input.
- **DIAG** is L1r: the whole trunk with ten recycles, read against a control case rather than a fixed bar.
- **The SKIP** is boltz2's L3, which is gated by injection instead.
- **The 5 LOOSE** are confidence tails at corr ≥ 0.99994:
  - in the protenix lineage, `torch.cdist`'s float32 expansion on the vendor side. Given an exact distance, they collapse to about 1e-4;
  - in chai1, a bfloat16 head.

  The whole-model harness grades these heads PARITY or CLOSE on every case.

## Fold accuracy

These numbers measure how well a model folds, not how faithfully it was ported. The table is 6MRR best-of-5 CA-RMSD, de novo from a single sequence (68 aa), 2026-09-14:

| model | Å | | model | Å |
|---|---|---|---|---|
| `boltz2` | 0.459 | | `intellifold2` | 1.512 |
| `alphafold3` | 0.628 | | `esmfold2_lm600m` | 1.522 |
| `opendde` | 0.729 | | `openfold3` | 1.547 |
| `rosettafold3` | 0.967 | | `protenix1` | 1.555 |
| `protenix2` | 1.016 | | `openbind0` | 1.578 |
| `esmfold2_fast` | 1.182 | | `esmfold2_lm300m` | 1.687 |
| `esmfold2` | 1.330 | | `chai1` | 1.713 |

AF2 runs single-seed at float32 with zero recycles: `af2_ptm` 1.712, `af2_multimer` 1.788.

Read these numbers as a band, not a ranking. Best-of-5 is a tail statistic, and some models vary run to run on the same seed: rosettafold3 seed 0 gave 0.968, 0.949 and 1.026. For a real comparison, use means over 20 or more samples.

Modality, from the L6 run:

| model | protein+ligand (Å / LDDT) | complex | RNA | DNA | confidence (mean / r) |
|---|---|---|---|---|---|
| `intellifold2` | 0.301 / 0.881 | 0.92 | 1.621 | 1.95 | 90.4 / .776 |
| `boltz2` | 0.385 / 0.907 | 0.92 | 1.419 | 1.59 | 96.9 / .830 |
| `protenix2` | 2.497 / 1.147 | 1.02 | 2.062 | 2.21 | 92.7 / .807 |
| `rosettafold3` | 0.464 / 0.889 | 1.12 | 1.143 | 2.67 | 85.2 / .708 |
| `opendde` | 1.269 / 0.870 | 1.49 | 1.372 | 1.99 | 89.9 / .635 |
| `chai1` | 0.335 / 0.993 | 1.57 | 1.648 | 1.89 | 78.9 / .543 |
| `openfold3` | 0.348 / 0.894 | 1.60 | 1.441 | 2.22 | 83.2 / .678 |

`r` is the correlation between predicted and actual error. A constant head would read r ≈ 0, whatever its mean.

## Re-running

```bash
# one model, all cases (the vendor's environment is looked up in VENDORS)
~/venv/bin/python dev/oracles/vendor_native_parity.py boltz2
# chosen cases; the full sampler; padded like a real run
~/venv/bin/python dev/oracles/vendor_native_parity.py rosettafold3 paired_msa template
~/venv/bin/python dev/oracles/vendor_native_parity.py protenix2 protein --trajectory
OURS_BUCKET=shipped ~/venv/bin/python dev/oracles/vendor_native_parity.py chai1 protein heterodimer
# alphafold3 against DeepMind's repository (~/af3_upstream, ~/venv_af3)
~/venv/bin/python dev/oracles/af3_native_parity.py
```

Each case runs in its own process, so that a case which runs out of memory does not starve the next. Useful environment variables:

| variable | effect |
|---|---|
| `VENDOR_NO_TAPS=1` | skip the vendor's localisation taps (large files) |
| `OPENDDE_CHUNK=64` | chunk OpenDDE's trunk, needed for `long_chain` |
| `TRAJ_FLOOR=trunk` | measure the vendor against itself with its trunk nudged 1e-5 |
| `MODEL_DIR` | use an alternative blob for our side |

chai1 also needs `CHAI_FP32=1` and the stock-chai interpreter in `CHAI_VENDOR_PY`.

Reference conformers are RDKit embeddings, so both sides must use the same RDKit version; `af3_native_parity.py` refuses to compare when they differ.

# CHECKPOINT: rosettafold3 vendor-native parity (paused 2026-10-19)

## Files
- `dev/oracles/rosettafold3_upstream_runner.py`: NEW, complete. It runs stock RF3 through
  `RF3InferenceEngine` (the CLI's engine args) and `engine.run(<rf3.json path>)`. The only
  override is `SampleDiffusion.sample_diffusion_like_af3`, which returns the injected coords
  and makes one real denoiser call. Precision choices: TF32 off, `trainer precision 32-true`,
  hard-coded `force_bfloat16` off for fp32, `DISABLE_CUEQUIVARIANCE=1`. It translates the AF3
  JSON into RF3 components: an MG ligand becomes a CCD ligand, a query-only MSA gets no
  msa_path, and a mod is written inline as `(CCD)`. `forward` asserts that its ref_pos draw
  equals the one in layout.npz.
- `dev/oracles/vendor_native_parity.py`: EDITED, minimal and commented.
  (a) A VENDORS row was added for `rosettafold3`.
  (b) `_s_inputs_to_ours` uses `converters.rosettafold3._AF3_TO_RF3_AATYPE` for
  rosettafold3, because atomworks swaps G/C and DG/DC relative to OF3.
- Nothing in src/ was touched. Nothing was committed.

## Env
- `~/venv_rf3`: uv, python 3.12. It was installed with `uv pip sync ~/venv_rf3/requirements-rf3.txt`,
  which was compiled from `~/venv_rf3/requirements.in` (foundry pyproject deps) plus
  `constraints.txt` (foundry ci-constraints: torch 2.7.1+cu126, rdkit 2025.3.6), with atomworks 2.2.1.
- The `rf3` extra (cuequivariance) is skipped.
- CCD_MIRROR_PATH and PDB_MIRROR_PATH are unset, so atomworks uses biotite's CCD. The MACE
  atom-level-embedding cache (IPD /net path) is absent, the same as for any external user.
- Checkpoint: `~/rf3_weights/rf3_foundry_01_24_latest_remapped.ckpt`.

## Commands that work
    cd ~/alphafold3-parity && JAX_DEFAULT_MATMUL_PRECISION=highest PYTHONPATH=src:. ~/venv/bin/python dev/oracles/vendor_native_parity.py rosettafold3 [CASES] --recycles 0
- Each case takes about 2 min.
- `B2_KEEP=/tmp/claude-1000/rf3keep` keeps the outputs. Note that the kept `rf3.json` has a
  temp msa_path, so fix it with sed before re-running the runner there.
- Localisation ablation scripts (vendor-side, NOT gates) are in the session scratchpad:
  `rf3_ablate.py` (rep atom -> CA) and `rf3_ablate2.py` (MSA dup-query and is_paired=0).

## Results, --recycles 0 (all 14 cases ran; every one FAILs)
In every case, s_inputs, trunk, distogram and the denoise step are at PARITY (1e-5), except
where the rows below say otherwise.

| case | worst row | verdict |
|---|---|---|
| protein | pae 0.11 | FAIL |
| msa | trunk pair 1.73 | FAIL |
| homodimer | pde 0.17 | FAIL |
| heterodimer | pde 0.19 | FAIL |
| rna | plddt corr 0.43 | FAIL |
| rna_mod | plddt corr 0.68 | FAIL |
| dna_mod | plddt corr 0.76 | FAIL |
| dna | plddt corr 0.87 | FAIL |
| protein_dna | plddt corr 0.96 | FAIL |
| ligand_ccd | s_inputs 2.45 | FAIL |
| ligand_smiles | s_inputs 3.23 | FAIL |
| ion | s_inputs 3.73 | FAIL |
| ptm | pde 0.19 | FAIL |
| covalent | token count 32 vs 39 | FAIL |

--recycles 1 (protein, msa) has NOT been run yet.

## Localised port divergences (all proven with vendor-side ablations or feature dumps)

1. **Confidence distance uses CA instead of the AF3 representative atom.**
   - Location: `src/alphafold3/model/network/confidence_head.py:272-281` uses
     `dense_atom_positions[:,1]`. RF3 uses `rep_atom_idxs` from
     `atomworks get_af3_token_representative_idxs`, which is CB (CA for Gly, C4/C2 for nucleics).
   - Evidence: the vendor run with rep atoms set to CA vs ours gives pae 0.114 -> 3.0e-3 and
     pde 0.164 -> 5.9e-3 (protein).
   - Fix: bin `positions` (the pseudo-beta gather, already computed above) instead of `ca`.

2. **Per-atom pLDDT logit indexing.**
   - RF3's logit k is the k-th REAL atom of the token, in atom_array order
     (AddIsRealAtom fills the first n slots).
   - Ours takes logit index = dense slot. For nucleics, dense slot 0 is the dropped OP3, so
     every atom reads its neighbour's logit.
   - Evidence: on rna, re-pairing by rank gives plddt corr 0.43 -> 0.9996.
   - Fix: at `confidence_head.py:~611` (and the experimentally_resolved head), for rosettafold3
     gather logits at `rank = cumsum(atom_mask)-1`.

3. **A ligand token's profile and MSA should be UNK (20), not GAP (21).**
   - Ion, ligand_ccd and ligand_smiles: the only s_inputs difference is the profile on ligand
     tokens (ours 21, vendor 20). The atom block is at 4e-6.
   - Cause: `chiral_features.apply_restype_alignment_on_atomized` (`chiral_features.py:374`)
     tests "gap in EVERY msa row", but padded rows are 0, so the test never fires.
   - Fix: restrict the test to live rows (`msa_mask.any(-1)`).

4. **MSA case.**
   - (a) We carry the query twice (5 rows); atomworks keeps the a3m as given (4 rows).
   - (b) When an alignment is loaded, RF3 sets is_paired=1 on the whole query row
     (`msa_stack[0,0,:,34]=1`). It is 0 only for the query-only path. Ours is 0 always
     (`evoformer.py:718`).
   - Evidence: the vendor run with the query duplicated and is_paired=0 reproduces ours at
     PARITY: pair 1.4e-5, single 3.7e-6, distogram 4.3e-6.
   - Fix: in the registry, set rosettafold3 `dedupe_self_msa='query'` (`_dedupe_query_row`).
     In `evoformer.py`, set `query_paired=1` for rosettafold3 when the chain has a real
     alignment (more than 1 live row).

5. **Covalent.**
   - atomworks atomises the bonded ASN 9 (its token count is +7 vs ours 32).
   - This is not yet localised further. Check `model_features` `declared_polymer_bonds='ligand'`
     and atomworks' rule for atomising residues in non-canonical inter-chain bonds.

## Next step
1. Run `--recycles 1` on protein and msa.
2. Localise the covalent atomisation rule (dump vendor feats with the scratchpad
   `dump_feats.py` and compare its atomised-token set).
3. Re-check ptm and dna_mod alignment.
4. Write the final report.

## Open questions
- None blocking.
- pLDDT for a vendor nucleic is written by RF3 itself in atom_array order (b-factor), so
  divergence 2 is a port bug, not a vendor quirk.

# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Tests for the per-model featurisation conventions.

Every convention here is SILENT when it is wrong: the batch stays the right
shape, the model runs, and the fold is merely worse. So each is pinned against
what the source model actually does rather than against a previous run.
"""

import numpy as np

from absl.testing import absltest
from alphafold3.model import model_registry
from alphafold3.model.pipeline import model_features


def _atom_window_batch(num_subsets=6, q_size=32, k_size=128, n_real=None):
  """A minimal batch carrying only the atom-attention gathers."""
  n_padded = num_subsets * q_size
  n_real = n_padded if n_real is None else n_real
  flat = np.arange(n_padded)
  mask = flat < n_real
  return {
      'token_atoms_to_queries:gather_idxs': flat.reshape(num_subsets, q_size),
      'token_atoms_to_queries:gather_mask': mask.reshape(num_subsets, q_size),
      'queries_to_keys:gather_idxs': np.zeros((num_subsets, k_size), np.int32),
      'queries_to_keys:gather_mask': np.zeros((num_subsets, k_size), bool),
      'tokens_to_queries:gather_idxs': (flat // 4).reshape(num_subsets, q_size),
      'tokens_to_queries:gather_mask': mask.reshape(num_subsets, q_size),
      'tokens_to_keys:gather_idxs': np.zeros((num_subsets, k_size), np.int32),
      'tokens_to_keys:gather_mask': np.zeros((num_subsets, k_size), bool),
  }


class KeyWindowTest(absltest.TestCase):

  def test_padding_masks_the_edge_blocks_instead_of_sliding_them(self):
    """opendde/protenix pad the key window; AF3 slides it back in bounds.

    Both give each block of 32 queries a 128-key window running from
    block_start - 48 to block_start + 79. AF3 shifts a window that runs off
    either end bodily back inside the atom count, so every block keeps 128 valid
    keys. opendde zero-pads instead and masks those slots, so its edge blocks
    genuinely see fewer neighbours -- and the ones they do see sit at DIFFERENT
    key slots, so the per-slot pair bias is misaligned too.
    """
    batch = model_features._padded_key_window(_atom_window_batch())

    keep = np.asarray(batch['queries_to_keys:gather_mask'])
    # The first block's leading 48 keys are before atom 0, the last block's
    # trailing 48 are past the end. Interior blocks are untouched.
    self.assertEqual(keep[0, :48].sum(), 0)
    self.assertTrue(keep[0, 48:].all())
    self.assertEqual(keep[-1, -48:].sum(), 0)
    for block in range(2, 4):
      self.assertTrue(keep[block].all(), f'interior block {block} lost keys')

  def test_padding_never_points_a_key_outside_the_atoms(self):
    batch = model_features._padded_key_window(_atom_window_batch())
    idxs = np.asarray(batch['queries_to_keys:gather_idxs'])
    self.assertTrue((idxs >= 0).all())
    self.assertTrue((idxs < 6 * 32).all())

  def test_padding_masks_atoms_that_are_only_padding(self):
    # A window slot may be in bounds and still be padding: the batch is padded
    # up to whole blocks, and those atoms are not real.
    batch = model_features._padded_key_window(_atom_window_batch(n_real=100))
    idxs = np.asarray(batch['queries_to_keys:gather_idxs'])
    keep = np.asarray(batch['queries_to_keys:gather_mask'])
    self.assertFalse(keep[idxs >= 100].any())

  def test_the_circular_window_wraps_rather_than_clipping(self):
    """chai-1 takes the window modulo the padded atom count.

    So the first block's leading 48 keys land at the END of the atom list, not
    at atom 0 -- which is where they land if you clip, and where AF3's slide
    would put real neighbours.
    """
    batch = model_features._circular_key_window(_atom_window_batch())
    idxs = np.asarray(batch['queries_to_keys:gather_idxs'])
    n_padded = 6 * 32
    np.testing.assert_array_equal(idxs[0, :48],
                                  np.arange(n_padded - 48, n_padded))
    self.assertTrue((idxs >= 0).all() and (idxs < n_padded).all())

  def test_the_two_conventions_disagree_only_at_the_edges(self):
    padded = model_features._padded_key_window(_atom_window_batch())
    circular = model_features._circular_key_window(_atom_window_batch())
    p = np.asarray(padded['queries_to_keys:gather_idxs'])
    c = np.asarray(circular['queries_to_keys:gather_idxs'])
    # The window reaches 48 atoms back and 79 forward, so with 32 queries per
    # block the first TWO and last TWO blocks run off the end; everything
    # between them is in bounds and the conventions cannot differ there.
    np.testing.assert_array_equal(p[2:-2], c[2:-2])
    self.assertFalse(np.array_equal(p[0], c[0]))
    self.assertFalse(np.array_equal(p[1], c[1]))


class ApplyTest(absltest.TestCase):

  def test_a_model_with_no_conventions_is_untouched(self):
    # openfold3 used to be in this list and is not any more: it declares
    # `centre_conformers` (model_config.CENTRE_REF_CONFORMERS). That knob had to
    # become a real featurise entry precisely BECAUSE openfold3 had none --
    # every caller guards the whole step on `if spec.featurise:`, so an empty
    # entry meant the convention never ran for the model it mattered most for.
    # ...and intellifold2 is not in it any more either, for the same reason: it
    # declares `qblock_keys` (its atom key window DUPLICATES the edge windows,
    # aligned to the padded query-block count). This list is a standing
    # invitation to rot -- every model that gains a convention has to leave it --
    # so it holds only the reference now.
    for name in ('alphafold3',):
      with self.subTest(model=name):
        before = _atom_window_batch()
        after = model_features.apply(_atom_window_batch(),
                                     model_registry.get(name))
        for key, value in before.items():
          np.testing.assert_array_equal(np.asarray(after[key]), value)

  def test_opendde_without_a_refeaturise_callable_is_an_error(self):
    # Rather than silently skipping the structural-token batch its diffusion
    # runs on, which would fold the residue tokens and look almost right.
    with self.assertRaises(ValueError):
      model_features.apply(_atom_window_batch(), model_registry.get('opendde'))


class _Chain:

  def __init__(self, chain_id, ptms=()):
    self.id = chain_id
    self.ptms = list(ptms)


class _FoldInput:

  def __init__(self, chains):
    self.chains = chains


def _token_batch(asym_id, residue_index, aatype=None):
  asym_id = np.asarray(asym_id)
  aatype = np.asarray(aatype if aatype is not None else np.zeros_like(asym_id))
  # The fields every featurised batch carries and the modified-residue
  # conventions read (749f425): chain-type flags, and a query-only MSA whose
  # profile is the query's one-hot.
  return {
      'asym_id': asym_id,
      'residue_index': np.asarray(residue_index),
      'aatype': aatype,
      'is_rna': np.zeros_like(asym_id),
      'is_dna': np.zeros_like(asym_id),
      'msa': aatype[None].copy(),
      'msa_mask': np.ones((1,) + aatype.shape, np.float32),
      'profile': np.eye(31, dtype=np.float32)[aatype],
  }


class CyclicPeriodTest(absltest.TestCase):
  """Cyclic is not a per-model convention: it wraps the SHARED encoding."""

  def test_every_chain_wraps_at_its_own_residue_count(self):
    batch = _token_batch([1, 1, 1, 2, 2, 0], [1, 2, 3, 1, 2, 0])
    out = model_features.cyclic_period(batch)['cyclic_period']
    np.testing.assert_array_equal(out, [3, 3, 3, 2, 2, 0])

  def test_the_period_is_residues_not_tokens(self):
    """A ligand contributes many tokens at ONE residue index.

    Wrapping on the token count would misplace every polymer offset in the same
    chain, and nothing would error.
    """
    batch = _token_batch([1, 1, 1, 1], [1, 2, 3, 3])
    out = model_features.cyclic_period(batch)['cyclic_period']
    np.testing.assert_array_equal(out, [3, 3, 3, 3])

  def test_naming_chains_leaves_the_others_linear(self):
    batch = _token_batch([1, 1, 2, 2], [1, 2, 1, 2])
    fold_input = _FoldInput([_Chain('A'), _Chain('B')])
    out = model_features.cyclic_period(batch, ['B'],
                                       fold_input=fold_input)['cyclic_period']
    np.testing.assert_array_equal(out, [0, 0, 2, 2])

  def test_an_unknown_chain_id_is_an_error(self):
    batch = _token_batch([1, 1], [1, 2])
    with self.assertRaises(ValueError):
      model_features.cyclic_period(batch, ['Z'],
                                   fold_input=_FoldInput([_Chain('A')]))


class ModifiedResidueTest(absltest.TestCase):
  """Boltz-2 reads a modified residue as one token, UNK, and a flag -- all three.

  Measured on phospho-ubiquitin: the restype alone leaves the phosphate 6.42 A
  out, the flag alone 6.53 A, the two together 3.10 A.
  """

  def test_the_flag_and_the_restype_land_on_the_modified_residue(self):
    batch = _token_batch([1, 1, 1, 2], [1, 2, 3, 1], aatype=[5, 6, 7, 8])
    fold_input = _FoldInput([_Chain('A', [('PTR', 2)]), _Chain('B')])
    out = model_features._mark_modified_residues(batch, fold_input,
                                                 unknown_restype=True)
    np.testing.assert_array_equal(out['is_modified'], [0, 1, 0, 0])
    self.assertEqual(int(out['aatype'][1]), 20)   # UNK, not the parent restype
    np.testing.assert_array_equal(out['aatype'][[0, 2, 3]], [5, 7, 8])

  def test_every_token_of_an_atomised_residue_is_marked(self):
    """Under AF3's convention one residue is several tokens sharing one index.

    So the tokens have to be found by (chain, residue), not by position.
    """
    batch = _token_batch([1, 1, 1, 1], [1, 2, 2, 3], aatype=[5, 6, 6, 7])
    out = model_features._mark_modified_residues(
        batch, _FoldInput([_Chain('A', [('PTR', 2)])]))
    np.testing.assert_array_equal(out['is_modified'], [0, 1, 1, 0])

  def test_the_restype_change_is_opt_in(self):
    # Every family except boltz2 keeps AF3's convention, the PARENT restype.
    batch = _token_batch([1, 1], [1, 2], aatype=[5, 6])
    out = model_features._mark_modified_residues(
        batch, _FoldInput([_Chain('A', [('PTR', 2)])]))
    np.testing.assert_array_equal(out['aatype'], [5, 6])
    np.testing.assert_array_equal(out['is_modified'], [0, 1])


class EmptyTemplateGapTest(absltest.TestCase):

  def test_atoms_mode_gaps_every_slot_and_token_without_atoms(self):
    # openfold3 starts every (slot, token) as GAP: a real template keeps what it covers, the rest reads 21
    aatype = np.zeros((4, 5), np.int32)
    aatype[0, :3] = [3, 7, 21]
    mask = np.zeros((4, 5, 24), np.float32)
    mask[0, :2, :4] = 1
    batch = {'template_aatype': aatype, 'template_atom_mask': mask}
    model_features._empty_template_gap(batch, slots='atoms')
    want = np.full((4, 5), 21)
    want[0, :2] = [3, 7]
    np.testing.assert_array_equal(batch['template_aatype'], want)

  def test_atoms_mode_matches_all_without_templates(self):
    batches = [{'template_aatype': np.zeros((4, 3), np.int32), 'template_atom_mask': np.zeros((4, 3, 24))}
               for _ in range(2)]
    model_features._empty_template_gap(batches[0], slots='all')
    model_features._empty_template_gap(batches[1], slots='atoms')
    np.testing.assert_array_equal(batches[0]['template_aatype'], batches[1]['template_aatype'])


class ChaiRowOrderTest(absltest.TestCase):

  def test_np126_argsort_matches_numpy_126(self):
    # numpy 1.26.4 (chai-lab's venv, a CPU without AVX-512): argsort of 72
    # bfd_uniclust priorities and the query's, the order chai read a 73-row
    # aligned.pqt in (vendor_native_parity chai1 paired_msa)
    order = model_features._np126_argsort([2] * 72 + [-1])
    self.assertEqual(order[:12], [72, 51, 50, 49, 48, 47, 46, 52, 45, 43, 42, 41])
    self.assertEqual(sorted(order), list(range(73)))

  def test_np126_argsort_sorts(self):
    rng = np.random.default_rng(0)
    for n in (1, 2, 15, 16, 17, 100, 1000):
      v = rng.integers(-1, 4, size=n)
      order = model_features._np126_argsort(v.tolist())
      np.testing.assert_array_equal(v[order], np.sort(v))


def _constraint_input(constraints):
  import json
  from alphafold3.common import folding_input
  return folding_input.Input.from_json(json.dumps({
      'dialect': 'alphafold3', 'version': 4, 'name': 't', 'modelSeeds': [1], 'constraints': constraints,
      'sequences': [{'protein': {'id': 'A', 'sequence': 'MKVLA'}},
                    {'ligand': {'id': 'L', 'ccdCodes': ['ATP']}}]}))


def _constraint_batch():
  """Tokens A1..A5 (A3 two tokens, as a modified residue might be), then three ligand atom tokens, then padding."""
  import types
  rows = [('A', 1, ['N', 'CA']), ('A', 2, ['CA']), ('A', 3, ['CA']), ('A', 3, ['SG']), ('A', 4, ['CA']),
          ('A', 5, ['CA']), ('L', 1, ['PG']), ('L', 1, ['O1G']), ('L', 1, ['PB']), ('', 0, [])]
  num, width = len(rows), 2
  chain, res, name = (np.full((num, width), '', object), np.zeros((num, width), int),
                      np.full((num, width), '', object))
  mask = np.zeros((num, width), bool)
  for t, (c, r, names) in enumerate(rows):
    for a, n in enumerate(names):
      chain[t, a], res[t, a], name[t, a], mask[t, a] = c, r, n, True
  return {'ref_mask': mask, 'token_atoms_layout': types.SimpleNamespace(chain_id=chain, res_id=res, atom_name=name)}


class ConstraintTest(absltest.TestCase):

  def test_boltz2_pocket_then_contact(self):
    # featurizerv2.py inference branch: UNSELECTED everywhere; pocket marks every token of a pocket residue against
    # the whole binder both ways; a contact marks only the FIRST token of each end and overwrites the pocket
    fi = _constraint_input([
        {'pocket': {'binder': 'L', 'contacts': [['A', 3]], 'maxDistance': 6.0}},
        {'contact': {'token1': ['A', 3], 'token2': ['L', 1, 'PB'], 'maxDistance': 9.0}}])
    b = model_features._boltz2_constraints(_constraint_batch(), fi)
    cls, thr = b['contact_conditioning'], b['contact_threshold']
    np.testing.assert_array_equal(cls[2, 6:9], [2, 2, 4])     # A3's first token: PG, O1G pocket; PB the contact
    np.testing.assert_array_equal(cls[3, 6:9], [2, 2, 2])     # A3's second token: pocket only
    np.testing.assert_array_equal(cls[6:9, 3], [3, 3, 3])     # binder -> pocket
    self.assertEqual(cls[8, 2], 4)
    self.assertEqual(thr[2, 8], 9.0)
    self.assertEqual(thr[3, 6], 6.0)
    self.assertEqual(cls[0, 1], 1)                            # untouched pairs UNSELECTED
    self.assertTrue((cls[9] == -1).all() and (cls[:, 9] == -1).all())

  def test_boltz2_names(self):
    with self.assertRaisesRegex(ValueError, 'named by its atom'):
      model_features._boltz2_constraints(_constraint_batch(), _constraint_input(
          [{'contact': {'token1': ['A', 1], 'token2': ['L', 1], 'maxDistance': 8.0}}]))
    with self.assertRaisesRegex(ValueError, 'names no token'):
      model_features._boltz2_constraints(_constraint_batch(), _constraint_input(
          [{'contact': {'token1': ['A', 1], 'token2': ['L', 1, 'C8'], 'maxDistance': 8.0}}]))

  def test_chai1_directed_contact_and_pocket(self):
    # token_dist_restraint.py: [token1, token2] only; token_pair_pocket_restraint.py: [pocket residue, every binder
    # token]; -1 everywhere else
    fi = _constraint_input([
        {'contact': {'token1': ['A', 1], 'token2': ['A', 5], 'maxDistance': 8.0}},
        {'pocket': {'binder': 'L', 'contacts': [['A', 2]], 'maxDistance': 6.0}}])
    b = model_features._chai1_constraints(_constraint_batch(), fi)
    contact, pocket = b['token_distance_restraint'], b['token_pair_pocket_restraint']
    self.assertEqual(contact[0, 5], 8.0)
    self.assertEqual(contact[5, 0], -1.0)
    self.assertEqual((contact != -1).sum(), 1)
    np.testing.assert_array_equal(pocket[1, 6:9], [6.0] * 3)
    self.assertEqual((pocket != -1).sum(), 3)

  def test_chai1_refuses_what_chai_lab_would_drop(self):
    for spec, msg in ((['A', 3], 'exactly one token'), (['L', 1, 'PG'], 'not atoms'), (['A', 1, 'CA'], 'not atoms')):
      with self.assertRaisesRegex(ValueError, msg):
        model_features._chai1_constraints(_constraint_batch(), _constraint_input(
            [{'contact': {'token1': ['A', 2], 'token2': spec, 'maxDistance': 8.0}}]))
    with self.assertRaisesRegex(ValueError, 'protein residues only'):
      model_features._chai1_constraints(_constraint_batch(), _constraint_input(
          [{'contact': {'token1': ['A', 2], 'token2': ['L', 1], 'maxDistance': 8.0}}]))

  def test_untrained_models_refuse(self):
    pocket = _constraint_input([{'pocket': {'binder': 'L', 'contacts': [['A', 3]], 'maxDistance': 6.0}}])
    contact = _constraint_input([{'contact': {'token1': ['A', 1], 'token2': ['A', 5], 'maxDistance': 8.0}}])
    for name in ('alphafold3', 'protenix2', 'openfold3', 'mol1'):
      with self.assertRaisesRegex(ValueError, 'not trained'):
        model_features.check_constraints_supported(contact, model_registry.get(name))
    model_features.check_constraints_supported(pocket, model_registry.get('boltz2'))
    model_features.check_constraints_supported(contact, model_registry.get('bioemu2'))
    with self.assertRaisesRegex(ValueError, 'contact conditioning only'):
      model_features.check_constraints_supported(pocket, model_registry.get('bioemu2'))

  def test_json_round_trip(self):
    from alphafold3.common import folding_input
    fi = _constraint_input([{'pocket': {'binder': 'L', 'contacts': [['A', 3]], 'maxDistance': 6.0}},
                            {'contact': {'token1': ['A', 1], 'token2': ['L', 1, 'PG'], 'maxDistance': 8}}])
    self.assertEqual(folding_input.Input.from_json(fi.to_json()).constraints, fi.constraints)
    with self.assertRaisesRegex(ValueError, 'beyond chain'):
      _constraint_input([{'contact': {'token1': ['A', 6], 'token2': ['A', 1], 'maxDistance': 8.0}}])


if __name__ == '__main__':
  absltest.main()

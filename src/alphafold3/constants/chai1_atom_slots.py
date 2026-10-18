"""chai-1's pLDDT slot per atom (chai_lab data/dataset/structure/utils.atom_37_atom_indices, Apache-2.0): per restype
in our vocabulary (residue_names.POLYMER_TYPES_WITH_UNKNOWN_AND_GAP) the atom name at each of the 37 slots; amino acids
share ATOM37, each nucleotide has its own order. None: chai indexes that residue's atoms by position (UNK) or, for a
single-atom token, slot 0."""

from alphafold3.constants import atom_types

SLOT_NAMES = (
    *(atom_types.ATOM37,) * 20,  # ALA .. VAL
    None,  # UNK
    None,  # -
    ("O4'", "C1'", "C2'", 'OP1', 'P', 'OP2', "O5'", "C5'", "C4'", "C3'", "O3'", "O2'", 'N1', 'C2', 'N3', 'C4', 'C5', 'C6', 'N6', 'N7', 'C8', 'N9', '', "H5'", "H5''", "H4'", "H3'", "H2'", "HO2'", "H1'", 'H2', 'H61', 'H62', 'H8', '', '', ''),  # A
    ("O4'", "C1'", "C2'", 'OP1', 'P', 'OP2', "O5'", "C5'", "C4'", "C3'", "O3'", "O2'", 'N1', 'C2', 'N2', 'N3', 'C4', 'C5', 'C6', 'O6', 'N7', 'C8', 'N9', "H5'", "H5''", "H4'", "H3'", "H2'", "HO2'", "H1'", 'H1', 'H22', 'H21', 'H8', '', '', ''),  # G
    ("O4'", "C1'", "C2'", 'OP1', 'P', 'OP2', "O5'", "C5'", "C4'", "C3'", "O3'", "O2'", 'N1', 'C2', 'O2', 'N3', 'C4', 'N4', 'C5', 'C6', '', '', '', "H5'", "H5''", "H4'", "H3'", "H2'", "HO2'", "H1'", 'H42', 'H41', 'H5', 'H6', '', '', ''),  # C
    ("O4'", "C1'", "C2'", 'OP1', 'P', 'OP2', "O5'", "C5'", "C4'", "C3'", "O3'", "O2'", 'N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C6', '', '', '', "H5'", "H5''", "H4'", "H3'", "H2'", "HO2'", "H1'", 'H3', 'H5', 'H6', '', '', '', ''),  # U
    ("O4'", "C1'", "C2'", 'OP1', 'P', 'OP2', "O5'", "C5'", "C4'", "C3'", "O3'", 'N9', 'C4', 'N3', 'C2', 'N1', 'C6', 'C5', 'N7', 'C8', 'N6', '', '', "H5''", "H5'", "H4'", "H3'", "H2''", "H2'", "H1'", 'H2', 'H61', 'H62', 'H8', '', '', ''),  # DA
    ("O4'", "C1'", "C2'", 'OP1', 'P', 'OP2', "O5'", "C5'", "C4'", "C3'", "O3'", 'N9', 'C4', 'N3', 'C2', 'N1', 'C6', 'C5', 'N7', 'C8', 'N2', 'O6', '', "H5''", "H5'", "H4'", "H3'", "H2''", "H2'", "H1'", 'H1', 'H22', 'H21', 'H8', '', '', ''),  # DG
    ("O4'", "C1'", "C2'", 'OP1', 'P', 'OP2', "O5'", "C5'", "C4'", "C3'", "O3'", 'N1', 'C2', 'O2', 'N3', 'C4', 'N4', 'C5', 'C6', '', '', '', '', "H5''", "H5'", "H4'", "H3'", "H2''", "H2'", "H1'", 'H42', 'H41', 'H5', 'H6', '', '', ''),  # DC
    ("O4'", "C1'", "C2'", 'OP1', 'P', 'OP2', "O5'", "C5'", "C4'", "C3'", "O3'", 'N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C7', 'C6', '', '', '', "H5''", "H5'", "H4'", "H3'", "H2''", "H2'", "H1'", 'H3', 'H71', 'H72', 'H73', 'H6', '', ''),  # DT
    None,  # N
)

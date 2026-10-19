import json
import warnings
from collections import namedtuple
from contextlib import contextmanager
from functools import cached_property, lru_cache

import numpy as np
from Bio.PDB import PDBIO, PDBParser, NeighborSearch, Superimposer, vectors  # vectors.calc_dihedral

from .data import unprocessed_atoms, atomic_types, at_types_lookup

# Atom type suffixes are appended to the atom type: "<type>/<tautomer or charge>", e.g. "CE1/HID", "NZ/+1".
# A rule is applied when all (protonated=True) or none (protonated=False) of the given hydrogens are present
# in the residue; the suffix is then added to the types of all listed atoms.
ChargeRule = namedtuple("ChargeRule", "resname trigger hydrogens protonated suffix atoms")
CHARGE_RULES = [
    ChargeRule("LYS", "NZ", ("HZ1", "HZ2", "HZ3"), True, "+1", ("NZ", "HZ1", "HZ2", "HZ3")),
    ChargeRule("ASP", "OD2", ("HD2",), False, "-1", ("OD2", "OD1", "CG")),
    ChargeRule("GLU", "OE2", ("HE2",), False, "-1", ("OE2", "OE1", "CD")),
    ChargeRule("CYS", "SG", ("HG",), False, "-1", ("SG",)),
    ChargeRule("TYR", "OH", ("HH",), False, "-1", ("OH", "CZ", "CE1", "CE2", "CD1", "CD2")),
    ChargeRule("ARG", "CZ", ("HE", "HH11", "HH12", "HH21", "HH22"), True, "+1",
               ("CZ", "NE", "HE", "NH1", "NH2", "HH11", "HH12", "HH21", "HH22")),
    ChargeRule("HIS", "CE1", ("HD1", "HE2"), True, "+1", ("CE1", "CD2", "ND1", "NE2", "HD1", "HE2")),
]
HISTIDINE_RING = ("CE1", "CD2", "ND1", "NE2")
SS_BRIDGE_DISTANCE = 2.2  # Å

# One row of the measured geometry of an atom, see Protein.get_lengths_angles
Measurement = namedtuple("Measurement", "res_name res_id atom_name atom_type bond_length bond_angle dihedral_angle")


@contextmanager
def _quiet_biopython():
    """Suppress the numpy warning raised inside Biopython's internal coordinates."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*'where' used without 'out'.*")
        yield


@lru_cache(maxsize=None)
def _ancestors_with_offsets(res_name, base_atom_type, atom_name):
    """Ancestor atom names and residue offsets (0 = same residue, +1 = next, -1 = previous)."""
    names, offsets = [], []
    for ancestor in atomic_types[res_name][base_atom_type]["ancestors"]:
        name, _, offset = ancestor.partition("/")
        names.append(name.replace("x", atom_name[2]) if "x" in name else name)
        offsets.append(int(offset) if offset else 0)
    return tuple(names), tuple(offsets)


class Protein:
    def __init__(self, pdb_file, output=True):
        if output:
            print(f"Loading protein from {pdb_file}...", end="")
        self.pdb_file = pdb_file
        self.structure = PDBParser(QUIET=True).get_structure(id="structure", file=pdb_file)
        with _quiet_biopython():
            self.structure.atom_to_internal_coordinates()
        self.residues = list(self.structure.get_residues())

        # assign atomic types
        self.atom_types = {}  # atom -> atom type, only for atoms which are processed
        self._assign_basic_atomic_types()
        self._assign_histidine_types()
        self._assign_formal_charges()
        if output:
            print(" ok")

    @cached_property
    def original_structure(self):
        """Structure as loaded from the file; parsed only when needed (preoptimization)."""
        return PDBParser(QUIET=True).get_structure(id="structure", file=self.pdb_file)

    # ---------------------------------------------------------------- atom types
    def _assign_basic_atomic_types(self):
        sg_atoms = [atom for residue in self.residues if residue.resname == "CYS"
                    for atom in residue.get_atoms() if atom.name == "SG"]
        sg_search = NeighborSearch(sg_atoms) if sg_atoms else None
        first_residue, last_residue = self.residues[0], self.residues[-1]

        for residue in self.residues:
            res_types = at_types_lookup.get(residue.resname, {})
            for atom in residue.get_atoms():
                if atom.name in unprocessed_atoms:
                    continue
                if residue.resname == "CYS" and atom.name == "SG":  # skip -SS- bridges
                    if len(sg_search.search(center=atom.coord, radius=SS_BRIDGE_DISTANCE, level="A")) == 2:
                        continue
                if residue is first_residue and atom.name in ("H", "N"):
                    continue
                if residue is last_residue and atom.name in ("O", "C"):
                    continue
                atom_type = res_types.get(atom.name)
                if atom_type:
                    self.atom_types[atom] = atom_type

    def _add_suffix(self, residue, atom_names, suffix):
        for name in atom_names:
            if name in residue and residue[name] in self.atom_types:
                self.atom_types[residue[name]] += "/" + suffix

    def _assign_histidine_types(self):
        for residue in self.residues:
            if residue.resname != "HIS" or "CE1" not in residue or residue["CE1"] not in self.atom_types:
                continue
            protonated = [h for h in ("HD1", "HE2") if h in residue]
            if len(protonated) == 1:  # exactly one of the two hydrogens: HID or HIE
                self._add_suffix(residue, HISTIDINE_RING + (protonated[0],), "HI" + protonated[0][1])

    def _assign_formal_charges(self):
        for residue in self.residues:
            for rule in CHARGE_RULES:
                if residue.resname != rule.resname or rule.trigger not in residue \
                        or residue[rule.trigger] not in self.atom_types:
                    continue
                present = [h in residue for h in rule.hydrogens]
                if all(present) if rule.protonated else not any(present):
                    self._add_suffix(residue, rule.atoms, rule.suffix)

    # ---------------------------------------------------------------- ancestors
    def _base_type(self, atom):
        return self.atom_types[atom].split("/")[0]

    def _type_data(self, atom):
        return atomic_types[atom.get_parent().resname][self._base_type(atom)]

    def _get_ancestors_with_offsets(self, atom):
        return _ancestors_with_offsets(atom.get_parent().resname, self._base_type(atom), atom.name)

    def _get_ancestors(self, atom):
        """Ancestor atom names without residue offsets."""
        return self._get_ancestors_with_offsets(atom)[0]

    def _has_neighbor_ancestors(self, atom):
        return any(self._get_ancestors_with_offsets(atom)[1])

    def _get_ancestor_atoms(self, res_i, atom):
        """Ancestor atoms, taken from the previous / next residue where the offset says so."""
        names, offsets = self._get_ancestors_with_offsets(atom)
        return [self.residues[res_i + offset][name] for name, offset in zip(names, offsets)]

    # ---------------------------------------------------------------- preoptimization
    def preoptimize(self, parameters_file):
        print("Setting bond lengths, bond angles and dihedral angles... ", end="")
        with open(parameters_file, "r", encoding="utf-8") as f:
            parameters = json.load(f)

        self._set_internal_coordinates(parameters)
        with _quiet_biopython():
            self.structure.internal_to_atom_coordinates()
        self._superimpose_on_original()

        self._place_single_bonded_atoms(parameters)
        # C and N depend on CA and on the neighbouring residue; O and H depend on the already placed C and N.
        # The atoms are placed residue by residue, so C(i) uses the not yet placed N(i+1), while N(i) uses
        # the already placed C(i-1). The result is therefore slightly asymmetric.
        self._place_atoms_with_neighbor_ancestors(parameters, ["C", "N"])
        self._place_atoms_with_neighbor_ancestors(parameters, ["O", "H"])
        print("ok")

    def _iter_typed_atoms(self, parameters):
        """Yield (residue index, residue, atom, atom parameters) for all atoms with an atom type."""
        for res_i, residue in enumerate(self.residues):
            for atom in residue.get_atoms():
                if atom in self.atom_types:
                    yield res_i, residue, atom, parameters[residue.resname][self.atom_types[atom]]

    def _set_internal_coordinates(self, parameters):
        """Set bond lengths, bond angles and dihedral angles of atoms whose ancestors are all in their own residue."""
        for res_i, residue, atom, atom_parameters in self._iter_typed_atoms(parameters):
            if self._has_neighbor_ancestors(atom):
                continue

            a2, a3, a4 = self._get_ancestors(atom)
            atom_type_data = self._type_data(atom)
            internal_coord = residue.internal_coord

            # bond length
            if internal_coord.get_length(f"{atom.name}:{a2}") is None:
                exit("error bond length")
            internal_coord.set_length(f"{atom.name}:{a2}", atom_parameters["bond_length"][0])

            # bond angle
            if atom_type_data.get("reverse order", False):
                angle_key = f"{atom.name}:{a2}:{a3}"
            else:
                angle_key = f"{a3}:{a2}:{atom.name}"
            if internal_coord.get_angle(angle_key) is None:
                exit("error bond angle")
            internal_coord.set_angle(angle_key, atom_parameters["bond_angle"][0])

            # dihedral angle (custom atoms are placed later, outside of the internal coordinates)
            if "dihedral_angle" in atom_parameters and "ancestors_type" not in atom_type_data:
                dihedral_key = f"{a4}:{a3}:{a2}:{atom.name}"
                current_dihedral = internal_coord.get_angle(dihedral_key)
                if current_dihedral is None:
                    exit("error dihedral angle")
                internal_coord.set_angle(dihedral_key,
                                         self._closest_cluster_center(current_dihedral,
                                                                      atom_parameters["dihedral_angle"]),
                                         overlap=False)

    def _superimpose_on_original(self):
        """Superimpose the structure with the new coordinates on the original one using CA atoms."""
        def alpha_carbons(structure):
            return [atom for atom in structure.get_atoms() if atom.element == "C" and atom.name == "CA"]

        sup = Superimposer()
        sup.set_atoms(alpha_carbons(self.original_structure), alpha_carbons(self.structure))
        with _quiet_biopython():
            sup.apply(self.structure)

    def _place_single_bonded_atoms(self, parameters):
        """Place custom atoms (atoms with a single chemical bond) using the closest cluster center of the dihedral."""
        for res_i, residue, atom, atom_parameters in self._iter_typed_atoms(parameters):
            if "dihedral_angle" not in atom_parameters or self._has_neighbor_ancestors(atom):
                continue
            if "ancestors_type" not in self._type_data(atom):
                continue

            A2, A3, A4 = self._get_ancestor_atoms(res_i, atom)
            current_dihedral = self._dihedral(atom, A2, A3, A4)
            atom.coord = self._compute_atom_position(
                A2.coord, A3.coord, A4.coord,
                atom_parameters["bond_length"][0], atom_parameters["bond_angle"][0],
                self._closest_cluster_center(current_dihedral, atom_parameters["dihedral_angle"]))

    def _place_atoms_with_neighbor_ancestors(self, parameters, atom_names):
        """Place atoms (C, N, O, H) having ancestors in the previous or the next residue."""
        for res_i, residue, atom, atom_parameters in self._iter_typed_atoms(parameters):
            if atom.name not in atom_names:
                continue

            A2, A3, A4 = self._get_ancestor_atoms(res_i, atom)
            if atom.name == "O":
                dihedral_angle = atom_parameters["dihedral_angle"][0]  # taken directly from the parameters
            else:
                dihedral_angle = self._dihedral(atom, A2, A3, A4)  # kept from the current geometry
            atom.coord = self._compute_atom_position(A2.coord, A3.coord, A4.coord,
                                                     atom_parameters["bond_length"][0],
                                                     atom_parameters["bond_angle"][0],
                                                     dihedral_angle)

    # ---------------------------------------------------------------- geometry
    @staticmethod
    def _dihedral(atom, a2, a3, a4):
        """Dihedral angle atom-a2-a3-a4 in degrees."""
        return np.degrees(vectors.calc_dihedral(atom.get_vector(), a2.get_vector(),
                                                a3.get_vector(), a4.get_vector()))

    @staticmethod
    def _closest_cluster_center(angle_deg, centers_deg):
        """Return the center (in degrees) closest to the angle, respecting the periodicity of angles."""
        diff = (np.deg2rad(angle_deg) - np.deg2rad(centers_deg) + np.pi) % (2 * np.pi) - np.pi
        return centers_deg[np.argmin(np.abs(diff))]

    @staticmethod
    def _compute_atom_position(a1, a2, a3, bond_length, bond_angle_deg, dihedral_angle_deg):
        """Compute the position of an atom bonded to a1 from its bond length, bond angle (atom-a1-a2)
        and dihedral angle (atom-a1-a2-a3). The sign of the dihedral angle is mirrored, because
        Biopython and vectors.calc_dihedral use the opposite handedness to the one used here."""
        a1 = np.array(a1, dtype=float)
        a2 = np.array(a2, dtype=float)
        a3 = np.array(a3, dtype=float)

        bond_angle = np.radians(bond_angle_deg)
        dihedral_angle = -np.radians(dihedral_angle_deg)

        v12 = a2 - a1
        v23 = a3 - a2

        # local orthonormal basis
        norm_v12 = np.linalg.norm(v12)
        if norm_v12 == 0:
            raise ValueError("Points a1 and a2 must not be identical.")
        e1 = v12 / norm_v12

        cross = np.cross(v12, v23)
        norm_cross = np.linalg.norm(cross)
        if norm_cross == 0:
            raise ValueError("Points a1, a2 and a3 are collinear.")
        e3 = cross / norm_cross

        e2 = np.cross(e3, e1)

        # local coordinates of the new atom
        x = bond_length * np.cos(bond_angle)
        y = bond_length * np.sin(bond_angle) * np.cos(dihedral_angle)
        z = bond_length * np.sin(bond_angle) * np.sin(dihedral_angle)

        # back to the global space
        return a1 + x * e1 + y * e2 + z * e3

    def save(self, output_PDB_file):
        print(f"Saving preoptimized protein to {output_PDB_file}... ", end="")
        io = PDBIO()
        io.set_structure(self.structure)
        io.save(output_PDB_file)
        print("ok\n")

    # ---------------------------------------------------------------- measurement
    def get_lengths_angles(self):
        """Return a Measurement (bond length, bond angle, dihedral angle) for every atom with an atom type."""
        measurements = []
        for res_i, residue in enumerate(self.residues):
            for atom, atom_type in ((atom, self.atom_types.get(atom)) for atom in residue.get_atoms()):
                if not atom_type:
                    continue
                a2, a3, a4 = self._get_ancestors(atom)
                A2, A3, A4 = self._get_ancestor_atoms(res_i, atom)
                atom_type_data = self._type_data(atom)
                internal_coord = residue.internal_coord

                bond_length = internal_coord.get_length(f"{atom.name}:{a2}")

                if A3.get_parent() is not residue:
                    bond_angle = np.degrees(vectors.calc_angle(atom.get_vector(), A2.get_vector(), A3.get_vector()))
                elif atom_type_data.get("reverse order", False):
                    bond_angle = internal_coord.get_angle(f"{atom.name}:{a2}:{a3}")
                else:
                    bond_angle = internal_coord.get_angle(f"{a3}:{a2}:{atom.name}")

                # internal coordinates are used only when all ancestors are in the same residue
                # (custom atoms and CB are computed from vectors)
                if "ancestors_type" in atom_type_data or atom.name == "CB" or self._has_neighbor_ancestors(atom):
                    dihedral_angle = self._dihedral(atom, A2, A3, A4)
                else:
                    dihedral_angle = internal_coord.get_angle(f"{a4}:{a3}:{a2}:{atom.name}")

                measurements.append(Measurement(residue.resname, residue.id[1], atom.name, atom_type,
                                                bond_length, bond_angle, dihedral_angle))
        return measurements

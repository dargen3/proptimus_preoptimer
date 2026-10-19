import numpy as np
import warnings
from Bio.PDB import PDBIO, PDBParser, NeighborSearch, Superimposer, vectors # vectors.calc_dihedral
from .data import unprocessed_atoms, atomic_types, at_types_lookup
import json


warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*'where' used without 'out'.*"
)


class Protein:
    def __init__(self,
                 pdb_file, output=True):
        if output:
            print(f"Loading protein from {pdb_file}...", end="")
        self.pdb_file = pdb_file
        self.original_structure = PDBParser(QUIET=True).get_structure(id="structure", file=pdb_file)
        self.structure = PDBParser(QUIET=True).get_structure(id="structure", file=pdb_file)
        self.structure.atom_to_internal_coordinates()
        self.residues = list(self.structure.get_residues())

        # assign atomic types
        self.kdtree = NeighborSearch(list(self.structure.get_atoms()))
        self._assign_basic_atomic_types()
        self._assign_histidine_types()
        self._assign_formal_charges()
        if output:
            print(" ok")

    def _assign_basic_atomic_types(self):
        for residue_i, residue in enumerate(self.residues):
            for atom in residue.get_atoms():
                if atom.name in unprocessed_atoms:
                    continue
                if residue.resname == "CYS" and atom.name == "SG":  # skip -SS- bridges
                    near_SG = [near_atom for near_atom in self.kdtree.search(center=atom.coord, radius=2.2, level="A")
                               if near_atom.name == "SG"]
                    if len(near_SG) == 2:
                        continue
                if atom.name == "H" and residue_i == 0:
                    continue
                if atom.name == "O" and residue is self.residues[-1]:
                    continue
                if atom.name == "C" and residue is self.residues[-1]:
                    continue
                if atom.name == "N" and residue_i == 0:
                    continue
                atom.atom_type = at_types_lookup[residue.resname][atom.name]

    def _assign_histidine_types(self):
        for residue in self.residues:
            for atom in residue.get_atoms():
                if not getattr(atom, "atom_type", False):
                    continue
                if residue.resname == "HIS" and atom.name == "CE1":
                    near_HH = [near_atom.name for near_atom in
                               self.kdtree.search(center=atom.coord, radius=4, level="A") if
                               residue == near_atom.get_parent() and near_atom.name in ["HD1", "HE2"]]
                    if near_HH == ["HD1"]:
                        atom.atom_type += "/HID"
                        residue["CD2"].atom_type += "/HID"
                        residue["ND1"].atom_type += "/HID"
                        residue["NE2"].atom_type += "/HID"
                        residue["HD1"].atom_type += "/HID"
                    elif near_HH == ["HE2"]:
                        atom.atom_type += "/HIE"
                        residue["CD2"].atom_type += "/HIE"
                        residue["ND1"].atom_type += "/HIE"
                        residue["NE2"].atom_type += "/HIE"
                        residue["HE2"].atom_type += "/HIE"

    def _assign_formal_charges(self):
        for residue in self.residues:
            for atom in residue.get_atoms():
                if not getattr(atom, "atom_type", False):
                    continue

                if residue.resname == "LYS" and atom.name == "NZ":
                    near_HZ = [near_atom for near_atom in self.kdtree.search(center=atom.coord, radius=2, level="A") if
                               residue == near_atom.get_parent() and near_atom.name[:2] == "HZ"]
                    if len(near_HZ) == 3:
                        atom.atom_type += "/+1"
                        for atom in near_HZ:
                            atom.atom_type += "/+1"

                if residue.resname == "ASP" and atom.name == "OD2":
                    near_HD2 = [near_atom for near_atom in self.kdtree.search(center=atom.coord, radius=2, level="A") if
                                residue == near_atom.get_parent() and near_atom.name == "HD2"]
                    if len(near_HD2) == 0:
                        atom.atom_type += "/-1"
                        residue["OD1"].atom_type += "/-1"
                        residue["CG"].atom_type += "/-1"

                if residue.resname == "GLU" and atom.name == "OE2":
                    near_HE2 = [near_atom for near_atom in self.kdtree.search(center=atom.coord, radius=2, level="A") if
                                residue == near_atom.get_parent() and near_atom.name == "HE2"]
                    if len(near_HE2) == 0:
                        atom.atom_type += "/-1"
                        residue["OE1"].atom_type += "/-1"
                        residue["CD"].atom_type += "/-1"

                if residue.resname == "CYS" and atom.name == "SG":
                    near_HG = [near_atom for near_atom in self.kdtree.search(center=atom.coord, radius=2, level="A") if
                               residue == near_atom.get_parent() and near_atom.name == "HG"]
                    if len(near_HG) == 0:
                        atom.atom_type += "/-1"

                if residue.resname == "TYR" and atom.name == "OH":  # možná pořešit i aromatický cyklus?
                    near_HH = [near_atom for near_atom in self.kdtree.search(center=atom.coord, radius=2, level="A") if
                               residue == near_atom.get_parent() and near_atom.name == "HH"]
                    if len(near_HH) == 0:
                        atom.atom_type += "/-1"
                        residue["CZ"].atom_type += "/-1"
                        residue["CE1"].atom_type += "/-1"
                        residue["CE2"].atom_type += "/-1"
                        residue["CD1"].atom_type += "/-1"
                        residue["CD2"].atom_type += "/-1"

                if residue.resname == "ARG" and atom.name == "CZ":
                    near_H = [near_atom for near_atom in self.kdtree.search(center=atom.coord, radius=4, level="A") if
                              residue == near_atom.get_parent() and near_atom.name in ["HE", "HH11", "HH12", "HH21",
                                                                                       "HH22"]]
                    if len(near_H) == 5:
                        atom.atom_type += "/+1"
                        residue["NE"].atom_type += "/+1"
                        residue["HE"].atom_type += "/+1"
                        residue["NH1"].atom_type += "/+1"
                        residue["NH2"].atom_type += "/+1"
                        residue["HH11"].atom_type += "/+1"
                        residue["HH12"].atom_type += "/+1"
                        residue["HH21"].atom_type += "/+1"
                        residue["HH22"].atom_type += "/+1"

                if residue.resname == "HIS" and atom.name == "CE1":
                    near_HH = [near_atom for near_atom in self.kdtree.search(center=atom.coord, radius=4, level="A") if
                               residue == near_atom.get_parent() and near_atom.name in ["HD1", "HE2"]]
                    if len(near_HH) == 2:
                        atom.atom_type += "/+1"
                        residue["CD2"].atom_type += "/+1"
                        residue["ND1"].atom_type += "/+1"
                        residue["NE2"].atom_type += "/+1"
                        try:
                            residue["HE2"].atom_type += "/+1"
                        except KeyError:
                            pass
                        try:
                            residue["HD1"].atom_type += "/+1"
                        except KeyError:
                            pass




    def preoptimize(self, parameters_file):
        print("Setting bond lengths, bond angles and dihedral angles... ", end="")
        with open(parameters_file, "r", encoding="utf-8") as f:
            parameters = json.load(f)

        self._set_internal_coordinates(parameters)
        self.structure.internal_to_atom_coordinates()
        self._superimpose_on_original()

        self._place_single_bonded_atoms(parameters)
        # C and N depend on CA and on the neighbouring residue; O and H depend on the already placed C and N
        self._place_atoms_with_neighbor_ancestors(parameters, ["C", "N"])
        self._place_atoms_with_neighbor_ancestors(parameters, ["O", "H"])
        print("ok")

    def _iter_typed_atoms(self, parameters):
        """Yield (residue index, residue, atom, atom parameters) for all atoms with an atom type."""
        for res_i, residue in enumerate(self.residues):
            for atom in residue.get_atoms():
                if getattr(atom, "atom_type", False):
                    yield res_i, residue, atom, parameters[residue.resname][atom.atom_type]

    def _set_internal_coordinates(self, parameters):
        """Set bond lengths, bond angles and dihedral angles of atoms whose ancestors are all in their own residue."""
        for res_i, residue, atom, atom_parameters in self._iter_typed_atoms(parameters):
            if self._has_neighbor_ancestors(atom):
                continue

            a2, a3, a4 = self._get_ancestors(atom)
            atom_type_data = atomic_types[residue.resname][atom.atom_type.split("/")[0]]
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
        sup = Superimposer()
        sup.set_atoms([atom for atom in self.original_structure.get_atoms() if atom.name == "CA"],
                      [atom for atom in self.structure.get_atoms() if atom.name == "CA"])
        sup.apply(self.structure)

    def _place_single_bonded_atoms(self, parameters):
        """Place custom atoms (atoms with a single chemical bond) using the closest cluster center of the dihedral."""
        for res_i, residue, atom, atom_parameters in self._iter_typed_atoms(parameters):
            if "dihedral_angle" not in atom_parameters or self._has_neighbor_ancestors(atom):
                continue
            if "ancestors_type" not in atomic_types[residue.resname][atom.atom_type.split("/")[0]]:
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

    def _get_ancestors_with_offsets(self, atom):
        """Ancestor atom names and residue offsets (0 = same residue, +1 = next, -1 = previous)."""
        names, offsets = [], []
        for ancestor in atomic_types[atom.get_parent().resname][atom.atom_type.split("/")[0]]["ancestors"]:
            name, _, offset = ancestor.partition("/")
            names.append(name.replace("x", atom.name[2]) if "x" in name else name)
            offsets.append(int(offset) if offset else 0)
        return names, offsets

    def _get_ancestors(self, atom):
        """Ancestor atom names without residue offsets."""
        a2, a3, a4 = self._get_ancestors_with_offsets(atom)[0]
        return a2, a3, a4

    def _has_neighbor_ancestors(self, atom):
        return any(self._get_ancestors_with_offsets(atom)[1])

    def _get_ancestor_atoms(self, res_i, atom):
        """Ancestor atoms, taken from the previous / next residue where the offset says so."""
        names, offsets = self._get_ancestors_with_offsets(atom)
        return [self.residues[res_i + offset][name] for name, offset in zip(names, offsets)]


    def get_lenghts_angles(self):

        for res_i, residue in enumerate(self.residues):

            res_name = residue.resname
            for atom in residue.get_atoms():
                atom_type = getattr(atom, "atom_type", False)
                if atom_type:


                    a2, a3, a4 = self._get_ancestors(atom)
                    A2, A3, A4 = self._get_ancestor_atoms(res_i, atom)
                    atom_type_data = atomic_types[res_name][atom_type.split("/")[0]]

                    # bond length
                    atom.bond_length = residue.internal_coord.get_length(f"{atom.name}:{a2}")

                    # bond angle
                    if A3.get_parent() is not residue:
                        atom.bond_angle = np.degrees(vectors.calc_angle(atom.get_vector(), A2.get_vector(), A3.get_vector()))
                    elif atom_type_data.get("reverse order", False):
                        atom.bond_angle = residue.internal_coord.get_angle(f"{atom.name}:{a2}:{a3}")
                    else:
                        atom.bond_angle = residue.internal_coord.get_angle(f"{a3}:{a2}:{atom.name}")

                    # dihedral angle
                    # internal coordinates are used only when all ancestors are in the same residue
                    # (custom atoms and CB are computed from vectors)
                    if "ancestors_type" in atom_type_data or atom.name == "CB" or self._has_neighbor_ancestors(atom):
                        atom.dihedral_angle = np.degrees(vectors.calc_dihedral(atom.get_vector(), A2.get_vector(), A3.get_vector(), A4.get_vector()))
                    else:
                        atom.dihedral_angle = residue.internal_coord.get_angle(f"{a4}:{a3}:{a2}:{atom.name}")


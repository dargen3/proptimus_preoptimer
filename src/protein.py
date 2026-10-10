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
                 pdb_file):
        self.pdb_file = pdb_file
        self.original_structure = PDBParser(QUIET=True).get_structure(id="structure", file=pdb_file)
        self.structure = PDBParser(QUIET=True).get_structure(id="structure", file=pdb_file)
        self.structure.atom_to_internal_coordinates()
        self.residues = list(self.structure.get_residues())

    def assign_atomic_types(self):
        self.kdtree = NeighborSearch(list(self.structure.get_atoms()))
        self._assign_basic_atomic_types()
        self._assign_histidine_types()
        self._assign_formal_charges()

    def preoptimize(self, parameters):
        self.assign_atomic_types()

        with open("optimized_proteins/parameters.json", "r", encoding="utf-8") as f:
            parameters = json.load(f)
        for residue in self.residues:
            for atom in residue.get_atoms():
                if not getattr(atom, "atom_type", False):
                    continue
                a2, a3, a4 = atomic_types[residue.resname][atom.atom_type.split("/")[0]]["ancestors"]
                atom_parameters = parameters[residue.resname][atom.atom_type]
                residue.internal_coord.set_length(f"{atom.name}:{a2}", atom_parameters["bond_length"][0])
                residue.internal_coord.set_angle(f"{atom.name}:{a2}:{a3}", atom_parameters["bond_angle"][0])
        self.structure.internal_to_atom_coordinates()
        sup = Superimposer()
        sup.set_atoms([atom for atom in self.original_structure.get_atoms() if atom.name == "CA"],
                      [atom for atom in self.structure.get_atoms() if atom.name == "CA"])
        sup.apply(self.structure)

    def save(self, output_PDB_file):
        io = PDBIO()
        io.set_structure(self.structure)
        io.save(output_PDB_file)

    def _assign_basic_atomic_types(self):
        for residue_i, residue in enumerate(self.residues):
            for atom in residue.get_atoms():
                if atom.name in unprocessed_atoms:
                    continue
                if residue.resname == "CYS" and atom.name == "SG": # skip -SS- bridges
                    near_SG = [near_atom for near_atom in self.kdtree.search(center=atom.coord, radius=2.2, level="A") if near_atom.name == "SG"]
                    if len(near_SG) == 2:
                        continue
                if atom.name == "H" and residue_i == 0:  # first residue, dodělat atomový typ
                    continue
                if atom.name == "O" and residue is self.residues[-1]:  # last residue, dodělat atomový typ
                    continue
                atom.atom_type = at_types_lookup[residue.resname][atom.name]


    def _assign_histidine_types(self):
        for residue in self.residues:
            for atom in residue.get_atoms():
                if residue.resname == "HIS" and atom.name == "CE1":
                    near_HH = [near_atom.name for near_atom in self.kdtree.search(center=atom.coord, radius=4, level="A") if
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

                if residue.resname == "GLU" and atom.name == "OE2":
                    near_HE2 = [near_atom for near_atom in self.kdtree.search(center=atom.coord, radius=2, level="A") if
                                  residue == near_atom.get_parent() and near_atom.name == "HE2"]
                    if len(near_HE2) == 0:
                        atom.atom_type += "/-1"
                        residue["OE1"].atom_type += "/-1"

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
                                  residue == near_atom.get_parent() and near_atom.name in ["HE", "HH11", "HH12", "HH21", "HH22"]]
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


    def get_lenghts_angles(self):





        for res_i, residue in enumerate(self.residues):
            # if residue.get_resname() == "TRP":
            #
            #     dostupne_uhly = residue.internal_coord.hedra.keys()
            #     for k in dostupne_uhly:
            #         nazvy_atomu = [atom_key.id.split("_")[-1] for atom_key in k]
            #         print(":".join(nazvy_atomu))
            #     #
            #     print("\n")
            #     dostupne_uhly = residue.internal_coord.dihedra.keys()
            #     for k in dostupne_uhly:
            #         nazvy_atomu = [atom_key.id.split("_")[-1] for atom_key in k]
            #         print(":".join(nazvy_atomu))
            #
            #     # print(dir(residue.internal_coord))
            #     # print(residue.internal_coord.dihedra)
            #     exit()
            res_name = residue.resname
            for atom in residue.get_atoms():
                atom_type = getattr(atom, "atom_type", False)
                if atom_type:

                    #if hasattr(atom, "group_formal_charge"):
                    #    at_name += f"{atom.group_formal_charge:+}"

                    #print(at_name)
                    #print(at_types_lookup["LYS"])

                    a2, a3, a4 = atomic_types[res_name][atom_type.split("/")[0]]["ancestors"]
                    if "x" in a2:
                        a2 = a2.replace("x", atom.name[2])
                    if "x" in a3:
                        a3 = a3.replace("x", atom.name[2])
                    if "x" in a4:
                        a4 = a4.replace("x", atom.name[2])
                    atom.bond_length = residue.internal_coord.get_length(f"{atom.name}:{a2}")
                    if atom.name in ["CB", "H", "HA", "HA2", "HA3", "O"]: # atoms directly bonded to backbone
                        if atom.name in ["CB", "O"]: # bacause of biopython
                            atom.bond_angle = residue.internal_coord.get_angle(f"{a3}:{a2}:{atom.name}")
                        else:
                            atom.bond_angle = residue.internal_coord.get_angle(f"{atom.name}:{a2}:{a3}")
                        if atom.name == "O":
                            atom.dihedral_angle = np.degrees(vectors.calc_dihedral(atom.get_vector(), residue[a2].get_vector(), residue[a3].get_vector(), self.residues[res_i+1][a4].get_vector()))
                        elif atom.name == "H":
                            atom.dihedral_angle = np.degrees(vectors.calc_dihedral(atom.get_vector(), residue[a2].get_vector(), residue[a3].get_vector(), self.residues[res_i-1][a4].get_vector()))
                        else:
                            atom.dihedral_angle = np.degrees(vectors.calc_dihedral(atom.get_vector(), residue[a2].get_vector(), residue[a3].get_vector(), residue[a4].get_vector()))
                    else:
                        atom.bond_angle = residue.internal_coord.get_angle(f"{a3}:{a2}:{atom.name}")
                        atom.dihedral_angle = residue.internal_coord.get_angle(f"{a4}:{a3}:{a2}:{atom.name}")


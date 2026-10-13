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


    #
    # def _assign_hydrogen_bonds(self):
    #     import biotite.structure as struc
    #     import biotite.structure.io as strucio
    #
    #     biotite_structure = strucio.load_structure(self.pdb_file,
    #                                                extra_fields=["charge"],
    #                                                include_bonds=True,
    #                                                model=1)
    #     hbonds = struc.hbond(biotite_structure, cutoff_dist=3.5)
    #     atoms = list(self.structure.get_atoms())
    #     for donor_i, hydrogen_i, acceptor_i in hbonds:
    #         atom = atoms[hydrogen_i]
    #         if not getattr(atom, "atom_type", False):
    #             continue
    #         if atom.atom_type[-7:] != "/hbondH":
    #             atom.atom_type += "/hbondH"

        # for donor_i, hydrogen_i, acceptor_i in hbonds:
        #     atom = atoms[donor_i]
        #     if not getattr(atom, "atom_type", False):
        #         continue
        #     if atom.atom_type[-7:] != "/hbondD":
        #         atom.atom_type += "/hbondD"
        #
        # for donor_i, hydrogen_i, acceptor_i in hbonds:
        #
        #     atom = atoms[acceptor_i]
        #     if not getattr(atom, "atom_type", False):
        #         continue
        #     if atom.atom_type[-7:] != "/hbondA":
        #         atom.atom_type += "/hbondA"






    def preoptimize(self, parameters):
        self.assign_atomic_types()

        with open("optimized_proteins/parameters.json", "r", encoding="utf-8") as f:
            parameters = json.load(f)
        for res_i, residue in enumerate(self.residues):
            for atom in residue.get_atoms():
                if not getattr(atom, "atom_type", False):
                    continue
                a2, a3, a4 = self._get_ancestors(atom)

                atom_parameters = parameters[residue.resname][atom.atom_type]





                # bond lengths
                test = residue.internal_coord.get_length(f"{atom.name}:{a2}")
                if test is None:
                    exit("error bond length")
                else:
                    residue.internal_coord.set_length(f"{atom.name}:{a2}", atom_parameters["bond_length"][0])

                # bond angles
                if atom.name in ["H", "HA", "HA2", "HA3"]:
                    test = residue.internal_coord.get_angle(f"{atom.name}:{a2}:{a3}")
                    if test is None:
                        exit("error bond angle")
                    else:
                        residue.internal_coord.set_angle(f"{atom.name}:{a2}:{a3}", atom_parameters["bond_angle"][0])
                else:
                    test = residue.internal_coord.get_angle(f"{a3}:{a2}:{atom.name}")
                    if test is None:
                        exit("error bond angle")
                    else:
                        residue.internal_coord.set_angle(f"{a3}:{a2}:{atom.name}", atom_parameters["bond_angle"][0])

                # dihedral angles
                def predict_cluster_from_centers(point_deg, centers_deg):
                    p_rad = np.deg2rad(point_deg)
                    c_rad = np.deg2rad(centers_deg)
                    diff = (p_rad - c_rad + np.pi) % (2 * np.pi) - np.pi
                    abs_diff = np.abs(diff)
                    closest_idx = np.argmin(abs_diff)
                    return  centers_deg[closest_idx]

                if "dihedral_angle" in atom_parameters:
                    dihedral_angle_parameters = atom_parameters["dihedral_angle"]

                    if "ancestors_type" not in atomic_types[residue.resname][atom.atom_type.split("/")[0]]:
                        test = residue.internal_coord.get_angle(f"{a4}:{a3}:{a2}:{atom.name}")
                        if test is None:
                            exit("error dihedral angle")
                        residue.internal_coord.set_angle(f"{a4}:{a3}:{a2}:{atom.name}",
                                                         predict_cluster_from_centers(test,
                                                                                      dihedral_angle_parameters),
                                                         overlap=False)






        self.structure.internal_to_atom_coordinates()
        sup = Superimposer()
        sup.set_atoms([atom for atom in self.original_structure.get_atoms() if atom.name == "CA"],
                      [atom for atom in self.structure.get_atoms() if atom.name == "CA"])
        sup.apply(self.structure)

        def vypocet_ctvrteho_bodu(a1, a2, a3, d, u, du, v_stupnich=True, invertovat_chiralitu=False):
            """
            Vypočítá souřadnice čtvrtého bodu a4.

            Parametry:
            - a1, a2, a3: souřadnice [x, y, z]
            - d: vzdálenost
            - u: vazebný úhel
            - du: dihedrální úhel
            - v_stupnich: True pro stupně, False pro radiány
            - invertovat_chiralitu: True pro zrcadlení výsledku (řeší opačnou orientaci)
            """
            a1 = np.array(a1, dtype=float)
            a2 = np.array(a2, dtype=float)
            a3 = np.array(a3, dtype=float)

            if v_stupnich:
                u = np.radians(u)
                du = np.radians(du)

            # Pokud je potřeba zrcadlit směr dihedrálního úhlu
            if invertovat_chiralitu:
                du = -du

            v12 = a2 - a1
            v23 = a3 - a2

            # 1. Lokální báze
            norm_v12 = np.linalg.norm(v12)
            if norm_v12 == 0:
                raise ValueError("Body a1 a a2 nemohou být totožné.")
            e1 = v12 / norm_v12

            cross = np.cross(v12, v23)
            norm_cross = np.linalg.norm(cross)
            if norm_cross == 0:
                raise ValueError("Body a1, a2 a a3 jsou kolineární.")
            e3 = cross / norm_cross

            e2 = np.cross(e3, e1)

            # 2. Lokální souřadnice bodu a4
            x = d * np.cos(u)
            y = d * np.sin(u) * np.cos(du)
            z = d * np.sin(u) * np.sin(du)

            # 3. Transformace zpět do globálního prostoru
            a4 = a1 + x * e1 + y * e2 + z * e3

            return a4

        for res_i, residue in enumerate(self.residues):
            for atom in residue.get_atoms():
                if not getattr(atom, "atom_type", False):
                    continue
                atom_parameters = parameters[residue.resname][atom.atom_type]

                if "dihedral_angle" in atom_parameters:
                    a2, a3, a4 = self._get_ancestors(atom)
                    if atom.atom_type.split("/")[0] == "O":
                        atom.coord = vypocet_ctvrteho_bodu(residue[a2].coord, residue[a3].coord,
                                                           self.residues[res_i + 1][a4].coord,
                                                           atom_parameters["bond_length"],
                                                           atom_parameters["bond_angle"],
                                                           atom_parameters["dihedral_angle"], invertovat_chiralitu=True)

                    else:

                        dihedral_angle_parameters = atom_parameters["dihedral_angle"]
                        if "ancestors_type" in atomic_types[residue.resname][atom.atom_type.split("/")[0]]:
                            test = np.degrees(vectors.calc_dihedral(atom.get_vector(), residue[a2].get_vector(), residue[a3].get_vector(), residue[a4].get_vector()))
                            ddd = predict_cluster_from_centers(test, dihedral_angle_parameters)
                            atom.coord = vypocet_ctvrteho_bodu(residue[a2].coord, residue[a3].coord, residue[a4].coord, atom_parameters["bond_length"], atom_parameters["bond_angle"], ddd, invertovat_chiralitu=True)






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

    def _get_ancestors(self, atom):
        a2, a3, a4 = atomic_types[atom.get_parent().resname][atom.atom_type.split("/")[0]]["ancestors"]
        if "x" in a2:
            a2 = a2.replace("x", atom.name[2])
        if "x" in a3:
            a3 = a3.replace("x", atom.name[2])
        return a2, a3, a4


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


                    a2, a3, a4 = self._get_ancestors(atom)

                    # bond length
                    atom.bond_length = residue.internal_coord.get_length(f"{atom.name}:{a2}")


                    # bond angle
                    if atom.name in ["CB", "H", "HA", "HA2", "HA3", "O"]: # atoms directly bonded to backbone
                        if atom.name in ["CB", "O"]: # bacause of biopython
                            atom.bond_angle = residue.internal_coord.get_angle(f"{a3}:{a2}:{atom.name}")
                        else:
                            atom.bond_angle = residue.internal_coord.get_angle(f"{atom.name}:{a2}:{a3}")
                    else:
                        atom.bond_angle = residue.internal_coord.get_angle(f"{a3}:{a2}:{atom.name}")

                    # dihedral angle
                    if "ancestors_type" in atomic_types[residue.resname][atom.atom_type.split("/")[0]]:
                        if atom.name == "O":
                            atom.dihedral_angle = np.degrees(
                                vectors.calc_dihedral(atom.get_vector(), residue[a2].get_vector(),
                                                      residue[a3].get_vector(),
                                                      self.residues[res_i + 1][a4].get_vector()))
                        else:
                            atom.dihedral_angle = np.degrees(vectors.calc_dihedral(atom.get_vector(), residue[a2].get_vector(), residue[a3].get_vector(), residue[a4].get_vector()))



                    else:
                        if atom.name in ["CB", "H", "HA", "HA2", "HA3", "O"]: # atoms directly bonded to backbone
                            if atom.name == "O":
                                atom.dihedral_angle = np.degrees(vectors.calc_dihedral(atom.get_vector(), residue[a2].get_vector(), residue[a3].get_vector(), self.residues[res_i+1][a4].get_vector()))
                            elif atom.name == "H":
                                atom.dihedral_angle = np.degrees(vectors.calc_dihedral(atom.get_vector(), residue[a2].get_vector(), residue[a3].get_vector(), self.residues[res_i-1][a4].get_vector()))
                            else:
                                atom.dihedral_angle = np.degrees(vectors.calc_dihedral(atom.get_vector(), residue[a2].get_vector(), residue[a3].get_vector(), residue[a4].get_vector()))
                        else:
                            atom.dihedral_angle = residue.internal_coord.get_angle(f"{a4}:{a3}:{a2}:{atom.name}")


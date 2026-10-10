import json
import numpy as np
from collections import defaultdict
from itertools import chain
from pathlib import Path
from sklearn.cluster import KMeans

from src.data import atomic_types
from src.graph import create_graph
from src.protein import Protein


def extract_structural_data(structures_dir): # smazat rrr
    structures_dir = Path(structures_dir)
    pdb_files = structures_dir.glob("*.pdb")
    preprocessed_data_json = structures_dir / Path("bond_lengths_angles.json")
    if preprocessed_data_json.exists():
        with open(preprocessed_data_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        proteins = [Protein(file) for file in list(pdb_files)[:5]]
        data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for i, protein in enumerate(proteins):
            protein.assign_atomic_types()
            protein.get_lenghts_angles()
            for residue in protein.residues:
                res_name = residue.resname
                res_id = residue.id[1]
                for atom in residue.get_atoms():
                    atom_type = getattr(atom, "atom_type", False)
                    if atom_type:
                        ID = f"{protein.pdb_file},{res_name}{res_id},{atom.name}"
                        atom_type_data = data[res_name][atom_type]
                        atom_type_data["ID"].append(ID)
                        atom_type_data["bond_length"].append(atom.bond_length)
                        atom_type_data["bond_angle"].append(atom.bond_angle)
                        atom_type_data["dihedral_angle"].append(atom.dihedral_angle)
        with open(preprocessed_data_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    return data


def run_kmeans(values, num_clusters):
    data = np.array(values, dtype=float).reshape(-1, 1)
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', n_init='auto', random_state=42)
    kmeans.fit(data)
    return kmeans.cluster_centers_.flatten().tolist()


def run_kmeans_angular(values, num_clusters):
    radians = np.deg2rad(values)
    x = np.cos(radians)
    y = np.sin(radians)
    data = np.column_stack((x, y))
    kmeans_instance = KMeans(n_clusters=num_clusters, init='k-means++', n_init=1)
    kmeans_instance.fit(data)
    centers_2d = kmeans_instance.cluster_centers_
    centers_deg = []
    for center in centers_2d:
        angle_rad = np.arctan2(center[1], center[0])
        centers_deg.append(np.rad2deg(angle_rad))
    return centers_deg

def calculate_parameters(structures_dir, data, at_types):
    structures_dir = Path(structures_dir)
    parameters_file = structures_dir / Path("parameters.json")
    if parameters_file.exists():
        with open(parameters_file, "r", encoding="utf-8") as f:
            parameters = json.load(f)
    else:
        parameters = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for residue_name, residue_at_types in data.items():
            for at_type, at_type_values in residue_at_types.items():
                bond_length_n_clusters, bond_angle_n_clusters, dihedral_angle_n_clusters = at_types[residue_name][at_type.split("/")[0]]["n_clusters"]
                if bond_length_n_clusters:
                    parameters[residue_name][at_type]["bond_length"] = run_kmeans(at_type_values["bond_length"], bond_length_n_clusters)
                if bond_angle_n_clusters:
                    parameters[residue_name][at_type]["bond_angle"] = run_kmeans(at_type_values["bond_angle"], bond_angle_n_clusters)
                if dihedral_angle_n_clusters:
                    parameters[residue_name][at_type]["dihedral_angle"] = run_kmeans_angular(at_type_values["dihedral_angle"], dihedral_angle_n_clusters)
        with open(parameters_file, "w", encoding="utf-8") as f:
            json.dump(parameters, f, ensure_ascii=False, indent=4)
    return parameters




res = "ALA"
structures_dir = "optimized_proteins"
data = extract_structural_data(structures_dir)

parameters = calculate_parameters(structures_dir, data, atomic_types)
for residue_name, residue_at_types in data.items():
    if residue_name != res: # smazat
        continue
    for atom_name, atom_values in residue_at_types.items():
        create_graph(data, parameters, residue_name, atom_name)



# vodíkové vazby
# náboj
# sterické vlivy



# upravit graf, aby ukazoval více informací (všechny atomy v atomovém typu a ancestory)
# udělat interaktivní bar s postupem extraktu dat a výpočtem parametrů



# outliers
# ALA H


# vodíky páteře (H) uvidíme po optimalizaci
# kyslíky páteře doplnit poté, co implementujeme rozdílné dihedrální atomy
# ALA HB - stérické vlivy
# SER HB - jdou udělat podle jiného dihedrálního úhlu (HB, CB, CA, OG)
# THR CG2, jdou udělat podle jiného dihedrálního úhlu (CG, CB, CA, OG1)
# THR HG - jsou tam píky - možná stérické vlivy?
# VAL HG - sterické vlivy
# LEU CD - sterické vlivy
# LEU HD - sterické vlivy
# LEU HB - sterické vlivy
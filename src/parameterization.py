import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from sklearn.cluster import KMeans
from tqdm import tqdm

from src.protein import Protein
from src.data import atomic_types


class Parameterization:
    def __init__(self, structures_dir, parameters_file):
        self.structures_dir = Path(structures_dir)
        self.parameters_file = parameters_file

        # Extracting structural data
        preprocessed_data_json = self.structures_dir / Path("bond_lengths_angles.json")
        pdb_files = sorted(self.structures_dir.glob("*.pdb"))[:5]
        self.data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for pdb_file in tqdm(pdb_files, desc="Extracting structural data", unit="protein"):
            protein = Protein(pdb_file, output=False)
            protein.get_lenghts_angles()
            for residue in protein.residues:
                res_name = residue.resname
                res_id = residue.id[1]
                for atom in residue.get_atoms():
                    atom_type = getattr(atom, "atom_type", False)
                    if atom_type:
                        ID = f"{protein.pdb_file},{res_name}{res_id},{atom.name}"
                        atom_type_data = self.data[res_name][atom_type]
                        atom_type_data["ID"].append(ID)
                        atom_type_data["bond_length"].append(atom.bond_length)
                        atom_type_data["bond_angle"].append(atom.bond_angle)
                        atom_type_data["dihedral_angle"].append(atom.dihedral_angle)
        with open(preprocessed_data_json, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)

        # Calculate parameters
        self.parameters = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        total = sum(len(at_types) for at_types in self.data.values())
        with tqdm(total=total, desc="Calculating parameters", unit="atom type") as progress:
            for residue_name, residue_at_types in self.data.items():
                for at_type, at_type_values in residue_at_types.items():
                    progress.set_postfix_str(f"{residue_name} {at_type}")
                    bond_length_n_clusters, bond_angle_n_clusters, dihedral_angle_n_clusters = atomic_types[residue_name][at_type.split("/")[0]]["n_clusters"]
                    if bond_length_n_clusters:
                        self.parameters[residue_name][at_type]["bond_length"] = self._run_kmeans(at_type_values["bond_length"], bond_length_n_clusters)
                    if bond_angle_n_clusters:
                        self.parameters[residue_name][at_type]["bond_angle"] = self._run_kmeans(at_type_values["bond_angle"], bond_angle_n_clusters)
                    if dihedral_angle_n_clusters:
                        self.parameters[residue_name][at_type]["dihedral_angle"] = self._run_kmeans(at_type_values["dihedral_angle"], dihedral_angle_n_clusters, angular_data=True)
                    progress.update()

        # Saving parameters
        with open(self.parameters_file, "w", encoding="utf-8") as f:
            json.dump(self.parameters, f, ensure_ascii=False, indent=4)


    @staticmethod
    def _run_kmeans(values, num_clusters, angular_data=False):
        # dihedral angles
        if angular_data:
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

        # 1D data, bond lengths and angles
        else:
            data = np.array(values, dtype=float).reshape(-1, 1)
            kmeans = KMeans(n_clusters=num_clusters, init='k-means++', n_init='auto', random_state=42)
            kmeans.fit(data)
            return kmeans.cluster_centers_.flatten().tolist()


    def find_outliers(self, gap_factor=0.5):
        print("\n\nOUTLIERS ANALYSIS:")
        outliers = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for residue_name, residue_atom_types in self.data.items():
            for atom_type, atom_type_values in residue_atom_types.items():
                for metric in ("bond_length", "bond_angle"):
                    if metric not in atom_type_values:
                        continue
                    values = np.array(atom_type_values[metric], dtype=float)
                    n = len(values)
                    ordered_indices = np.argsort(values)
                    sorted_values = values[ordered_indices]
                    bulk_width = np.percentile(sorted_values, 99.5) - np.percentile(sorted_values, 0.5)
                    flagged = np.zeros(n, dtype=bool)
                    for m in range(1, n // 100 + 1):
                        if sorted_values[n - m] - sorted_values[n - m - 1] > gap_factor * bulk_width:
                            flagged[n - m:] = True
                        if sorted_values[m] - sorted_values[m - 1] > gap_factor * bulk_width:
                            flagged[:m] = True
                    outliers_mask = np.zeros(n, dtype=bool)
                    outliers_mask[ordered_indices[flagged]] = True
                    if not outliers_mask.any():
                        continue
                    center = self.parameters[residue_name][atom_type][metric][0]
                    deviations = values - center
                    distances_in_md = np.abs(deviations) / np.nanmean(np.abs(deviations[~outliers_mask]))
                    for i in np.flatnonzero(outliers_mask):
                        pdb_file, residue, atom_name = atom_type_values["ID"][i].rsplit(",", 2)
                        res_id = int(residue[3:])
                        outliers[pdb_file][(res_id, residue_name)][atom_name].append((metric, values[i], center, distances_in_md[i]))

        units = {"bond_length": "Å", "bond_angle": "°"}
        for pdb_file in sorted(outliers):
            print(f"\n{pdb_file}")
            for (res_id, residue_name), atoms in sorted(outliers[pdb_file].items()):
                print(f"    {residue_name}{res_id}")
                for atom_name, atom_outliers in sorted(atoms.items()):
                    for metric, value, center, distance_in_md in sorted(set(atom_outliers)):
                        unit = units[metric]
                        print(f"        {atom_name}: {metric} = {value:.3f}{unit}, estimated value: {center:.3f}{unit}, {distance_in_md:.1f} MD")
        print("")

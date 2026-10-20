# PROPTIMUS PREOPTIMER

## Description

PROPTIMUS PREOPTIMER is a parameterizable tool for setting bond lengths, bond angles and dihedral angles in protein structures. 
In the `preoptimize` mode, the internal coordinates of the atoms are set according to the parameters. 
Optionally, only the positions of hydrogen atoms can be preoptimized while all other atoms are kept in place. 
In the `parameterize` mode, the parameters are calculated from a set of protein structures. 

The `parameters` directory currently contains parameters derived from structures optimised with the [GFN-FF](https://onlinelibrary.wiley.com/doi/10.1002/anie.202004239) force field.

## Getting Started
Before running PROPTIMUS PREOPTIMER for the first time, you need to create and activate a Python virtual environment where the script will always be executed. Python 3.11 or newer is required. You can set up the virtual environment in two steps:

**1. Clone GitHub repository:**

```
git clone https://github.com/sb-ncbr/proptimus_preoptimer
cd proptimus_preoptimer
```

**2. Set up a virtual environment**

* Create a virtual environment named `.venv`:

```
python3 -m venv .proptimus_preoptimer_venv
```

* Activate the virtual environment:

```
source .proptimus_preoptimer_venv/bin/activate
```

* Install the dependencies:

```
pip install -r requirements.txt
```

Once the virtual environment is created and activated, no further installation is required, and PROPTIMUS PREOPTIMER is ready to run.

## Executing the calculation

### Required arguments

`--mode {preoptimize,parameterize}`          Mode `preoptimize` sets bond lengths, bond angles and dihedral angles according to the parameters. Mode `parameterize` creates these parameters from a set of protein structures.

### Arguments for `--mode preoptimize`

`--input_PDB_file FILE`          PDB file with structure, which should be preoptimised (required).

`--output_PDB_file FILE`         PDB file to store preoptimized structure. The file must not exist (required).

`--parameters FILE`              JSON file with parameters. Available parameters are in the `parameters` directory (required).

`--only_hydrogens`               Preoptimize only hydrogen atoms, positions of all other atoms are kept.

### Arguments for `--mode parameterize`

`--proteins_directory DIR`       Directory with protein structures (`*.pdb` files) used to calculate the parameters (required).

`--data_dir DIR`                 Directory to store the results. The directory must not exist. The calculated parameters are saved to `<data_dir>/parameters.json` and HTML graphs of all atom types of all residues to the `<data_dir>/graphs` subdirectory (required).

`--workers N`                    Number of worker processes (default: number of CPUs).

`--open_browser`                 Open the HTML graphs in the default web browser when they are ready.

### Example of executing the calculation:

Preoptimization of a structure with the GFN-FF parameters:

```bash
python preoptimer.py --mode preoptimize --input_PDB_file structure.pdb --output_PDB_file structure_preoptimized.pdb --parameters parameters/GFN-FF_parameters.json
```

Preoptimization of hydrogen atoms only:

```bash
python preoptimer.py --mode preoptimize --input_PDB_file structure.pdb --output_PDB_file structure_preoptimized.pdb --parameters parameters/GFN-FF_parameters.json --only_hydrogens
```

Calculation of new parameters from a set of structures:

```bash
python preoptimer.py --mode parameterize --proteins_directory my_structures --data_dir my_parameterization --workers 8
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
